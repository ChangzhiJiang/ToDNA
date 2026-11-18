#!/usr/bin/env python3
"""Fine-tune a full 229M gReLU Enformer on one sealed Gosai half."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# gReLU 1.0.4 predates NumPy 2.x and still uses the removed alias.
if not hasattr(np, "product"):
    np.product = np.prod


TASK_COLUMNS = ("K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained-state-dict", type=Path, required=True)
    parser.add_argument("--role", choices=("design", "evaluation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="Visible CUDA device indices. Pass two indices for DDP training.",
    )
    parser.add_argument(
        "--distributed-backend",
        choices=("gloo", "nccl"),
        help="Explicit backend for multi-GPU DDP. Leave unset for Lightning's default.",
    )
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation fraction must be between zero and one half")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = pd.read_csv(
        args.table,
        sep="\t",
        usecols=("sequence",) + TASK_COLUMNS,
    )
    # DFSeqDataset interprets the first dataframe column as sequence and all
    # remaining columns as tasks; pandas preserves source, not usecols, order.
    frame = frame.loc[:, ["sequence", *TASK_COLUMNS]]
    if args.max_rows is not None:
        if args.max_rows < 2:
            raise ValueError("max rows must be at least two")
        frame = frame.iloc[: args.max_rows].copy()
    valid_dna = frame["sequence"].str.fullmatch("[ACGT]{200}")
    if not bool(valid_dna.all()):
        raise ValueError(f"table contains {int((~valid_dna).sum())} invalid 200 bp rows")
    if not np.isfinite(frame[list(TASK_COLUMNS)].to_numpy()).all():
        raise ValueError("table contains non-finite activity values")

    validation = frame.sample(frac=args.validation_fraction, random_state=args.seed)
    training = frame.drop(index=validation.index)
    training = training.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    from grelu.data.dataset import DFSeqDataset
    from grelu.lightning import LightningModel

    if args.distributed_backend is not None:
        if len(args.device) < 2:
            raise ValueError("--distributed-backend requires at least two devices")
        # gReLU 1.0.4 constructs its Trainer internally without exposing a
        # strategy argument. Patch that single construction point locally.
        import grelu.lightning as grelu_lightning
        from pytorch_lightning.strategies import DDPStrategy

        trainer_class = grelu_lightning.pl.Trainer

        def trainer_with_backend(*trainer_args, **trainer_kwargs):
            trainer_kwargs["strategy"] = DDPStrategy(
                process_group_backend=args.distributed_backend
            )
            return trainer_class(*trainer_args, **trainer_kwargs)

        grelu_lightning.pl.Trainer = trainer_with_backend

    train_dataset = DFSeqDataset(
        training,
        seq_len=200,
        rc=False,
        max_seq_shift=2,
        seed=args.seed,
        augment_mode="random",
    )
    validation_dataset = DFSeqDataset(validation, seq_len=200, rc=False)
    run_dir = args.output.parent / f"{args.output.stem}_training"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_params = {"model_type": "EnformerModel", "n_tasks": 3}
    train_params = {
        "task": "regression",
        "loss": "mse",
        "logger": "csv",
        "name": args.role,
        "lr": 1e-4,
        "optimizer": "adam",
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "devices": args.device,
        "num_workers": args.num_workers,
        "save_dir": str(run_dir),
        "checkpoint": {
            "dirpath": str(checkpoint_dir),
            "filename": f"{args.role}-epoch{{epoch:02d}}-val{{val_loss:.4f}}",
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 1,
            "save_last": True,
            "save_weights_only": True,
        },
    }
    model = LightningModel(model_params, train_params)
    if args.distributed_backend is not None:
        # gReLU 1.0.4 omits sync_dist on its metric logging. Synchronize both
        # scalar and dict logs so ModelCheckpoint sees the full validation set.
        model_log = model.log
        model_log_dict = model.log_dict

        def synchronized_log(*log_args, **log_kwargs):
            log_kwargs.setdefault("sync_dist", True)
            return model_log(*log_args, **log_kwargs)

        def synchronized_log_dict(*log_args, **log_kwargs):
            log_kwargs.setdefault("sync_dist", True)
            return model_log_dict(*log_args, **log_kwargs)

        model.log = synchronized_log
        model.log_dict = synchronized_log_dict
    pretrained = torch.load(
        args.pretrained_state_dict, map_location="cpu", weights_only=False
    )
    if isinstance(pretrained, dict) and "state_dict" in pretrained:
        pretrained = pretrained["state_dict"]
    if not isinstance(pretrained, dict):
        raise ValueError("pretrained artifact does not contain a state dictionary")
    embedding_state: dict[str, torch.Tensor] = {}
    for key, value in pretrained.items():
        if key.startswith("embedding."):
            embedding_state[key.removeprefix("embedding.")] = value
        elif key.startswith("model.embedding."):
            embedding_state[key.removeprefix("model.embedding.")] = value
    if not embedding_state:
        raise ValueError("pretrained artifact contains no Enformer embedding tensors")
    model.model.embedding.load_state_dict(embedding_state, strict=True)
    trainer = model.train_on_dataset(train_dataset, validation_dataset)
    if not trainer.is_global_zero:
        return
    best_path = Path(trainer.checkpoint_callback.best_model_path)
    if not best_path.is_file():
        raise RuntimeError(f"best checkpoint was not written: {best_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    try:
        os.link(best_path, args.output)
    except OSError:
        shutil.copy2(best_path, args.output)

    metadata = {
        "role": args.role,
        "protocol": "DNA-CRAFT-style disjoint-half full Enformer fine-tuning",
        "official_author_checkpoint": False,
        "author_split_seed_available": False,
        "source_table": str(args.table),
        "source_table_sha256": sha256(args.table),
        "source_rows": len(frame),
        "debug_max_rows": args.max_rows,
        "training_rows": len(training),
        "validation_rows": len(validation),
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "sequence_length": 200,
        "model_params": model_params,
        "pretrained_state_dict": str(args.pretrained_state_dict),
        "pretrained_state_dict_sha256": sha256(args.pretrained_state_dict),
        "train_params": {
            "epochs": args.epochs,
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "global_batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_sequence_shift": 2,
            "reverse_complement_augmentation": False,
            "distributed_backend": args.distributed_backend,
            "synchronize_distributed_metrics": args.distributed_backend is not None,
        },
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score),
        "checkpoint": str(args.output),
        "checkpoint_sha256": sha256(args.output),
        "checkpoint_size_bytes": args.output.stat().st_size,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
