#!/usr/bin/env python3
"""Train an Evo 2 Vortex LoRA adapter on natural ENCODE enhancer windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from evo2 import Evo2

from src.lora import (
    Evo2LoraConfig,
    adapter_state,
    inject_lora,
    iter_lora_modules,
    lora_parameters,
    materialize_autograd_tensors,
    restore_adapter_state,
    save_lora_adapter,
    set_lora_training,
    trainable_parameter_summary,
)
from src.sft import (
    causal_lm_losses,
    deterministic_orientation,
    perplexity,
    random_orientations,
    read_sequence_jsonl,
    reverse_complement,
    tokenize_sequences,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def batch_at(sequences: list[str], step: int, batch_size: int, rng: random.Random) -> list[str]:
    if batch_size > len(sequences):
        raise ValueError("batch size exceeds training corpus")
    indices = [rng.randrange(len(sequences)) for _ in range(batch_size)]
    return random_orientations([sequences[index] for index in indices], rng)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    sequences: list[str],
    device: torch.device | str,
    batch_size: int,
    max_sequences: int,
    seed: int,
    both_orientations: bool,
) -> dict[str, float | int]:
    set_lora_training(model, False)
    selected = sequences[:max_sequences]
    oriented = []
    for sequence in selected:
        if both_orientations:
            oriented.extend((sequence, reverse_complement(sequence)))
        else:
            oriented.append(deterministic_orientation(sequence, seed))
    sums = {"full_vocab_nll": 0.0, "dna_conditional_nll": 0.0}
    count = 0
    for start in range(0, len(oriented), batch_size):
        tokens = tokenize_sequences(oriented[start : start + batch_size], device)
        logits, _ = model(tokens)
        losses = causal_lm_losses(logits, tokens)
        batch_count = tokens.shape[0]
        for name in sums:
            sums[name] += float(losses[name]) * batch_count
        count += batch_count
    metrics = {name: value / count for name, value in sums.items()}
    metrics.update(
        {
            "full_vocab_perplexity": perplexity(metrics["full_vocab_nll"]),
            "dna_conditional_perplexity": perplexity(
                metrics["dna_conditional_nll"]
            ),
            "evaluated_sequences": count,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="evo2_7b")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--last-n-blocks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-sequences", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument(
        "--validation-both-orientations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.warmup_steps >= args.max_steps:
        raise ValueError("warmup steps must be smaller than max steps")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_root / "train.jsonl"
    validation_path = args.data_root / "validation.jsonl"
    train_sequences = read_sequence_jsonl(train_path)
    validation_sequences = read_sequence_jsonl(validation_path)
    data_hashes = {"train": sha256(train_path), "validation": sha256(validation_path)}
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    argument_config = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    progress_path = args.output_dir / "progress.json"
    atomic_json(progress_path, {"stage": "loading_evo2", "config": argument_config})
    evo = Evo2(args.model_name)
    model = evo.model
    materialize_autograd_tensors(model)
    if not 1 <= args.last_n_blocks <= len(model.blocks):
        raise ValueError("last-n-blocks must be within the Evo 2 block count")
    block_indices = tuple(range(len(model.blocks) - args.last_n_blocks, len(model.blocks)))
    config = Evo2LoraConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        block_indices=block_indices,
    )
    targets = inject_lora(model, config)
    set_lora_training(model, True)
    trainable = trainable_parameter_summary(model)
    device = model.block_idx_to_device[0]
    parameters = lora_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def lr_multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    rng = random.Random(args.seed)
    metadata = {
        "protocol": "Evo 2 natural-enhancer causal-LM SFT LoRA",
        "model": args.model_name,
        "config": argument_config,
        "data_sha256": data_hashes,
        "trainable": trainable,
        "target_modules": targets,
        "selection_metric": "validation full_vocab_nll",
        "test_evaluators_used_for_selection": False,
    }
    atomic_json(args.output_dir / "run_config.json", metadata)
    metrics_path = args.output_dir / "metrics.jsonl"
    best_nll = float("inf")
    start_step = 0
    baseline = None
    if args.resume_from is not None:
        resume = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if resume["data_sha256"] != data_hashes:
            raise ValueError("resume data hashes do not match current corpus")
        if resume["lora_config"] != {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "block_indices": block_indices,
        }:
            raise ValueError("resume LoRA configuration does not match arguments")
        restore_adapter_state(model, resume["adapter_state"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        rng.setstate(resume["python_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        if torch.cuda.is_available() and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        start_step = int(resume["step"])
        best_nll = float(resume["best_full_vocab_nll"])
        baseline = resume["baseline_validation"]
    started = time.time()

    if baseline is None:
        baseline = evaluate(
            model,
            validation_sequences,
            device,
            args.validation_batch_size,
            args.validation_sequences,
            args.seed,
            args.validation_both_orientations,
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": 0, "validation": baseline}) + "\n")
    set_lora_training(model, True)
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, args.max_steps + 1):
        running_loss = 0.0
        for _ in range(args.gradient_accumulation):
            batch = batch_at(train_sequences, step, args.batch_size, rng)
            tokens = tokenize_sequences(batch, device)
            logits, _ = model(tokens)
            losses = causal_lm_losses(logits, tokens)
            loss = losses["full_vocab_nll"] / args.gradient_accumulation
            loss.backward()
            running_loss += float(loss)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 10 == 0:
            print(
                f"TRAIN step={step}/{args.max_steps} loss={running_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} grad_norm={gradient_norm:.4f}",
                flush=True,
            )
        if step % args.validate_every == 0 or step == args.max_steps:
            validation = evaluate(
                model,
                validation_sequences,
                device,
                args.validation_batch_size,
                args.validation_sequences,
                args.seed,
                args.validation_both_orientations,
            )
            event = {
                "step": step,
                "elapsed_seconds": time.time() - started,
                "train_full_vocab_nll": running_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": gradient_norm,
                "validation": validation,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            improved = validation["full_vocab_nll"] < best_nll
            if improved:
                best_nll = float(validation["full_vocab_nll"])
                save_lora_adapter(
                    model,
                    args.output_dir / "best_adapter.pt",
                    config,
                    {**metadata, "best_step": step, "validation": validation},
                )
            save_lora_adapter(
                model,
                args.output_dir / "latest_adapter.pt",
                config,
                {**metadata, "step": step, "validation": validation},
            )
            atomic_torch_save(
                args.output_dir / "train_state.pt",
                {
                    "step": step,
                    "best_full_vocab_nll": best_nll,
                    "baseline_validation": baseline,
                    "data_sha256": data_hashes,
                    "lora_config": {
                        "rank": args.rank,
                        "alpha": args.alpha,
                        "dropout": args.dropout,
                        "block_indices": block_indices,
                    },
                    "adapter_state": adapter_state(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "python_rng_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    ),
                },
            )
            atomic_json(
                progress_path,
                {
                    "stage": "training",
                    "step": step,
                    "max_steps": args.max_steps,
                    "best_full_vocab_nll": best_nll,
                    "latest_validation": validation,
                    "elapsed_seconds": time.time() - started,
                },
            )
            print(f"VALIDATION step={step} metrics={json.dumps(validation)}", flush=True)
            set_lora_training(model, True)

    summary = {
        **metadata,
        "baseline_validation": baseline,
        "best_full_vocab_nll": best_nll,
        "best_adapter": str(args.output_dir / "best_adapter.pt"),
        "best_adapter_sha256": sha256(args.output_dir / "best_adapter.pt"),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "complete").touch()
    atomic_json(progress_path, {"stage": "complete", **summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
