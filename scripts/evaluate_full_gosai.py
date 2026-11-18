#!/usr/bin/env python3
"""Post-hoc scoring with the public full-data Gosai gReLU checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CELL_NAMES = ("K562", "HepG2", "SK-N-SH")

# gReLU 1.0.4 predates NumPy 2.x and still calls the removed alias.
if not hasattr(np, "product"):
    np.product = np.prod


def _as_activity_matrix(predictions: np.ndarray) -> np.ndarray:
    activity = np.asarray(predictions)
    if activity.ndim == 3 and activity.shape[-1] == 1:
        activity = activity[..., 0]
    if activity.ndim != 2 or activity.shape[1] != len(CELL_NAMES):
        raise ValueError(
            "expected gReLU predictions with shape [batch, 3] or [batch, 3, 1], "
            f"got {activity.shape}"
        )
    return activity


def _min_gap(activity: np.ndarray, target_index: int) -> np.ndarray:
    off_targets = [index for index in range(3) if index != target_index]
    return activity[:, target_index] - activity[:, off_targets].max(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    from grelu.data.dataset import SeqDataset
    from grelu.lightning import LightningModel

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    records = payload["candidates"]
    if not records:
        raise ValueError("candidate file is empty")

    initial_sequences = [record["initial_sequence"] for record in records]
    optimized_sequences = [record["optimized_sequence"] for record in records]
    if any(len(sequence) != payload["length"] for sequence in initial_sequences):
        raise ValueError("an initial sequence does not match the declared length")
    if any(len(sequence) != payload["length"] for sequence in optimized_sequences):
        raise ValueError("an optimized sequence does not match the declared length")

    # The public checkpoint contains the complete Enformer trunk and task head,
    # but its saved type is ``EnformerPretrainedModel``. That constructor first
    # downloads the generic 5313-task Enformer artifact from W&B. The local
    # ``EnformerModel`` has the identical architecture for this checkpoint and
    # lets Lightning restore all parameters without any network access.
    model = LightningModel.load_from_checkpoint(
        str(args.checkpoint),
        model_params={"model_type": "EnformerModel", "n_tasks": 3},
        map_location="cpu",
        weights_only=False,
    )
    dataset = SeqDataset(
        initial_sequences + optimized_sequences,
        seq_len=payload["length"],
        rc=False,
    )
    predictions = model.predict_on_dataset(
        dataset,
        devices=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    activity = _as_activity_matrix(predictions)
    split = len(records)
    initial_activity = activity[:split]
    optimized_activity = activity[split:]
    target_index = CELL_NAMES.index(payload["target"])
    initial_gap = _min_gap(initial_activity, target_index)
    optimized_gap = _min_gap(optimized_activity, target_index)

    for index, record in enumerate(records):
        record["full_gosai_evaluation"] = {
            "initial_activity": initial_activity[index].tolist(),
            "optimized_activity": optimized_activity[index].tolist(),
            "initial_mingap": float(initial_gap[index]),
            "optimized_mingap": float(optimized_gap[index]),
            "mingap_change": float(optimized_gap[index] - initial_gap[index]),
        }

    payload["full_gosai_checkpoint"] = str(args.checkpoint)
    payload["full_gosai_cell_order"] = CELL_NAMES
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
