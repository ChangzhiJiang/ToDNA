#!/usr/bin/env python3
"""Run locked LoRA Evo 2 + TTO on the frozen baseline's exact initial pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from evo2_nabla_dna import (
    Evo2TestTimeDesigner,
    MultiCellActivityReward,
    load_activity_checkpoint,
)
from run_full229m_hparam_search import CELL_NAMES, target_metrics, write_json
from run_search_sweep import build_search_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_pool(path: Path, expected_hash: str, expected_count: int) -> list[str]:
    if sha256(path) != expected_hash:
        raise ValueError("initial pool hash does not match frozen baseline")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = payload["sequences"]
    if len(sequences) != expected_count or len(set(sequences)) != expected_count:
        raise ValueError("initial pool count or uniqueness is invalid")
    if any(len(sequence) != 200 or set(sequence) - set("ACGT") for sequence in sequences):
        raise ValueError("initial pool contains invalid DNA")
    return sequences


def completed_result(path: Path, top_k: int) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload.get("top_candidates", [])) != top_k:
        raise ValueError(f"incomplete existing result: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-summary", type=Path, required=True)
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-lock-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.json"

    frozen_manifest = json.loads(
        (args.frozen_lock_root / "manifest.json").read_text(encoding="utf-8")
    )
    adapter_summary = json.loads(args.adapter_summary.read_text(encoding="utf-8"))
    adapter_hash = sha256(args.adapter)
    if adapter_hash != adapter_summary["best_adapter_sha256"]:
        raise ValueError("formal adapter hash does not match its training summary")
    pool_path = Path(frozen_manifest["pool_file"])
    pool_size = int(frozen_manifest["num_initial"])
    top_k = int(frozen_manifest["top_k"])
    seed = int(frozen_manifest["seed"])
    sequences = load_pool(pool_path, frozen_manifest["pool_sha256"], pool_size)
    config = dict(frozen_manifest["winner"]["config"])
    if json_sha256(config) != frozen_manifest["config_sha256"]:
        raise ValueError("frozen TTO configuration hash mismatch")

    write_json(progress, {"stage": "loading_lora_evo2"})
    designer = Evo2TestTimeDesigner(
        model_name="evo2_7b", lora_adapter=args.adapter
    )
    designer.dto.disable_flash_attention()
    write_json(progress, {"stage": "loading_design_model"})
    predictor = load_activity_checkpoint(
        str(args.design_checkpoint), device=designer.dto.device
    )
    design_hash = sha256(args.design_checkpoint)

    candidate_files = []
    for target_index, target in enumerate(CELL_NAMES):
        slug = target.lower().replace("-", "")
        result_path = args.output_root / "results" / f"{slug}.json"
        lock_path = args.output_root / "formal_lock" / f"{slug}_seed{seed}_top{top_k}.json"
        existing = completed_result(result_path, top_k)
        if existing is None:
            write_json(
                progress,
                {
                    "stage": "lora_tto_search",
                    "target": target,
                    "target_index": target_index,
                    "pool_size": pool_size,
                    "top_k": top_k,
                },
            )
            search_seed = seed * 1000 + target_index * 10 + 1
            random.seed(search_seed)
            torch.manual_seed(search_seed)
            reward = MultiCellActivityReward(
                predictor=predictor,
                target_index=target_index,
                off_target_indices=[i for i in range(3) if i != target_index],
                smooth_max_temperature=0.1,
            )
            records = designer.search(
                prompt="ACGT",
                length=200,
                reward_fn=reward,
                config=build_search_config(config, pool_size, pool_size),
                initial_sequences=sequences,
            )
            if len(records) < top_k:
                raise RuntimeError(f"{target}: only {len(records)} accepted candidates")
            ranking_penalty = float(config["ranking_prior_penalty"])
            records.sort(
                key=lambda record: record.optimized_reward
                - ranking_penalty * record.evo2_nll,
                reverse=True,
            )
            selected = records[:top_k]
            search_summary = dict(designer.last_search_summary)
            selected_payloads = []
            for record in selected:
                payload = record.to_dict()
                payload["lora_evo2_nll"] = record.evo2_nll
                selected_payloads.append(payload)
            existing = {
                "protocol": "lora_evo2_tto_formal_search",
                "selection_boundary": (
                    "Design-Model, LoRA Evo 2 prior, and hard constraints only"
                ),
                "target": target,
                "cell_order": list(CELL_NAMES),
                "seed": seed,
                "length": 200,
                "pool_size": pool_size,
                "top_k": top_k,
                "pool_file": str(pool_path),
                "pool_sha256": frozen_manifest["pool_sha256"],
                "config": config,
                "config_sha256": frozen_manifest["config_sha256"],
                "adapter": str(args.adapter),
                "adapter_sha256": adapter_hash,
                "design_checkpoint": str(args.design_checkpoint),
                "design_checkpoint_sha256": design_hash,
                "search_summary": search_summary,
                "selection_metrics_all_accepted": target_metrics(
                    records, search_summary
                ),
                "selection_metrics_top128": target_metrics(
                    selected,
                    {"attempted": pool_size, "accepted": search_summary["accepted"]},
                ),
                "top_candidates": selected_payloads,
            }
            write_json(result_path, existing)
        locked = {
            **{
                key: value
                for key, value in existing.items()
                if key not in {"top_candidates"}
            },
            "protocol": "lora_evo2_tto_formal_candidate_lock",
            "source_result": str(result_path),
            "candidates": existing["top_candidates"],
        }
        write_json(lock_path, locked)
        candidate_files.append(str(lock_path))
        print(f"LORA_TTO_TARGET_COMPLETE target={target} path={lock_path}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "protocol": "lora_evo2_tto_formal_candidate_lock",
        "paired_frozen_lock": str(args.frozen_lock_root),
        "winner": {
            "name": "formal_encode_sft_lora",
            "config": config,
            "adapter": str(args.adapter),
            "adapter_sha256": adapter_hash,
        },
        "seed": seed,
        "num_initial": pool_size,
        "top_k": top_k,
        "pool_file": str(pool_path),
        "pool_sha256": frozen_manifest["pool_sha256"],
        "config": config,
        "config_sha256": frozen_manifest["config_sha256"],
        "adapter": str(args.adapter),
        "adapter_sha256": adapter_hash,
        "design_checkpoint": str(args.design_checkpoint),
        "design_checkpoint_sha256": design_hash,
        "candidate_files": candidate_files,
        "evaluation_boundary": "Evaluation-Model and FullMPRA have not been used",
    }
    write_json(args.output_root / "formal_lock" / "manifest.json", manifest)
    (args.output_root / "formal_lock" / "complete").touch()
    (args.output_root / "formal_search.complete").touch()
    write_json(progress, {"stage": "formal_search_complete", "manifest": manifest})
    print("LORA_TTO_FORMAL_SEARCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
