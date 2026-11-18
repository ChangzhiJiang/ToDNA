#!/usr/bin/env python3
"""Independent post-lock FullMPRA scoring for ToDNA supplement locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "product"):
    np.product = np.prod


TARGETS = ("k562", "hepg2", "sknsh")
SEEDS = (20260920, 20260921, 20260922)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n_seeds": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def lock_specs(stage_root: Path) -> list[tuple[str, str, int, Path, list[str]]]:
    specs = []
    for arm_dir in sorted(stage_root.iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        for seed in SEEDS:
            for target in TARGETS:
                path = arm_dir / f"seed{seed}" / "formal_lock" / f"{target}_seed{seed}_top256.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = payload.get("candidates", [])
                sequences = [str(row.get("optimized_sequence", "")).upper() for row in rows]
                if len(sequences) != 256 or len(set(sequences)) != 256 or any(len(seq) != 200 or set(seq) - set("ACGT") for seq in sequences):
                    raise ValueError(f"invalid immutable lock: {path}")
                if payload.get("evaluation_model_used") is not False or payload.get("sealed_test_opened") is not False:
                    raise ValueError(f"evaluation/sealed boundary violation: {path}")
                specs.append((arm, target, seed, path, sequences))
    if len(specs) != 72:
        raise ValueError(f"expected 72 locks, found {len(specs)}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    specs = lock_specs(args.stage_root.resolve())
    sequences = [sequence for _, _, _, _, values in specs for sequence in values]

    from grelu.data.dataset import SeqDataset
    from grelu.lightning import LightningModel

    model = LightningModel.load_from_checkpoint(
        str(args.checkpoint),
        model_params={"model_type": "EnformerModel", "n_tasks": 3},
        map_location="cpu",
        weights_only=False,
    )
    predictions = np.asarray(
        model.predict_on_dataset(
            SeqDataset(sequences, seq_len=200, rc=False),
            devices=args.device,
            num_workers=0,
            batch_size=args.batch_size,
        )
    )
    if predictions.ndim == 3 and predictions.shape[-1] == 1:
        predictions = predictions[..., 0]
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError(f"unexpected FullMPRA prediction shape {predictions.shape}")

    records = []
    cursor = 0
    for arm, target, seed, path, values in specs:
        block = predictions[cursor : cursor + len(values)]
        cursor += len(values)
        target_index = TARGETS.index(target)
        off_targets = [index for index in range(3) if index != target_index]
        target_activity = block[:, target_index]
        min_gap = target_activity - block[:, off_targets].max(axis=1)
        records.append({
            "arm": arm,
            "target": target,
            "seed": seed,
            "n_sequences": len(values),
            "lock": str(path),
            "lock_sha256": sha256(path),
            "fullmpra_target_activity_mean": float(target_activity.mean()),
            "fullmpra_target_activity_sd_within_lock": float(target_activity.std(ddof=1)),
            "fullmpra_min_gap_mean": float(min_gap.mean()),
            "fullmpra_min_gap_sd_within_lock": float(min_gap.std(ddof=1)),
        })

    groups = defaultdict(list)
    for row in records:
        groups[(row["arm"], row["target"])].append(row)
    aggregate = []
    for (arm, target), rows in sorted(groups.items()):
        aggregate.append({
            "arm": arm,
            "target": target,
            "fullmpra_target_activity": describe([row["fullmpra_target_activity_mean"] for row in rows]),
            "fullmpra_min_gap": describe([row["fullmpra_min_gap_mean"] for row in rows]),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "per_seed.json", records)
    write_json(args.output / "aggregate.json", aggregate)
    manifest = {
        "protocol": "todna_supplements_fullmpra_v1",
        "analysis_unit": "reward-or-setpoint-arm x enhancer-target x seed",
        "rows": len(records),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "selection_changed": False,
        "evaluation_model_used": True,
        "sealed_test_opened": False,
        "source_locks": [{"path": row["lock"], "sha256": row["lock_sha256"]} for row in records],
    }
    write_json(args.output / "manifest.json", manifest)
    files = {name: sha256(args.output / name) for name in ("per_seed.json", "aggregate.json", "manifest.json")}
    write_json(args.output / "complete.json", {**manifest, "files": files})
    print(json.dumps({"rows": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
