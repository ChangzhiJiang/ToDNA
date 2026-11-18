#!/usr/bin/env python3
"""Post-hoc scoring with an evaluation-only activity predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src import load_activity_checkpoint


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
CELL_NAMES = ("K562", "HepG2", "SK-N-SH")


def one_hot(sequences: list[str], device: torch.device) -> torch.Tensor:
    indices = torch.tensor(
        [[BASE_TO_INDEX[base] for base in sequence] for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    return F.one_hot(indices, num_classes=4).float()


def min_gap(activity: torch.Tensor, target_index: int) -> torch.Tensor:
    off_targets = [index for index in range(3) if index != target_index]
    return activity[:, target_index] - activity[:, off_targets].max(dim=-1).values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--evaluation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    records = payload["candidates"]
    if not records:
        raise ValueError("candidate file is empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_activity_checkpoint(str(args.evaluation_checkpoint), device=device)
    initial_sequences = [record["initial_sequence"] for record in records]
    optimized_sequences = [record["optimized_sequence"] for record in records]
    target_index = CELL_NAMES.index(payload["target"])
    with torch.no_grad():
        initial_activity = model(one_hot(initial_sequences, device))
        optimized_activity = model(one_hot(optimized_sequences, device))
        initial_gap = min_gap(initial_activity, target_index)
        optimized_gap = min_gap(optimized_activity, target_index)

    for index, record in enumerate(records):
        record["evaluation"] = {
            "initial_activity": initial_activity[index].cpu().tolist(),
            "optimized_activity": optimized_activity[index].cpu().tolist(),
            "initial_mingap": float(initial_gap[index]),
            "optimized_mingap": float(optimized_gap[index]),
            "mingap_change": float(optimized_gap[index] - initial_gap[index]),
            "hamming_edits": sum(
                before != after
                for before, after in zip(initial_sequences[index], optimized_sequences[index])
            ),
        }
    payload["evaluation_checkpoint"] = str(args.evaluation_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
