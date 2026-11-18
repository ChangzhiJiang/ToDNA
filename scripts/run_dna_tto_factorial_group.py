#!/usr/bin/env python3
"""Run four factorial cells sharing one loaded Base or LoRA Evo2 model.

The loaded model is fixed for ``--init-context`` and reused for all four
pool/reference combinations. This preserves the factorial factors while
avoiding four identical 7B model loads. The shared model-load audit is written
once per seed; each cell keeps its own optimization audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from src import load_activity_checkpoint
from src.dto import DtoConfig
from src.design import SearchConfig
from src.profiling import CostProfiler
from run_dna_tto_factorial import (
    CONFIG, FactorialDesigner, load_cache, load_pool, load_starts, sha256,
    sequence_metrics, write_json,
)
from run_evo2_hparam_matched_seed import CtrlStyleDeltaR


LEVELS = ("base", "lora")


def cell_name(pool: str, init: str, reference: str) -> str:
    return f"pool_{pool}_init_{init}_ref_{reference}"


def run_cell(
    *,
    designer: FactorialDesigner,
    predictor,
    pool_path: Path,
    starts_path: Path,
    cache_path: Path,
    design_checkpoint: Path,
    adapter: Path,
    output_root: Path,
    seed: int,
    pool_context: str,
    init_context: str,
    reference_context: str,
    kl_weight: float,
) -> dict[str, object]:
    pool = load_pool(pool_path)
    starts = load_starts(starts_path, set(pool))
    trajectories = [sequence for sequence in starts for _ in range(12)]
    contexts = load_cache(cache_path, starts)
    profiler = CostProfiler(method="ToDNA", protocol="strict_hepg2_lora_pool_init_ref_factorial_v1")
    search_config = SearchConfig(
        num_initial=len(trajectories),
        top_k=256,
        dto=DtoConfig(
            steps=10, learning_rate=0.1, temperature=0.7, reward_weight=1.0,
            prior_weight=0.0, gradient_clip_norm=5.0, initial_sequence_margin=0.25,
            trust_region_weight=kl_weight, edit_penalty_weight=0.0, max_edits=None,
            initial_logit_noise_std=0.02,
            initial_logit_noise_seed=seed * 100000,
        ),
        constraints=None, prior_penalty=0.0,
    )
    reward = CtrlStyleDeltaR(predictor, "hepg2").to(designer.dto.device)
    random.seed(seed)
    torch.manual_seed(seed)
    records = designer.search_with_contexts(
        reward_fn=reward, config=search_config, trajectories=trajectories,
        contexts=contexts, init_context=init_context,
        reference_context=reference_context, profiler=profiler,
    )
    if len(records) != 256:
        raise RuntimeError(f"{cell_name(pool_context, init_context, reference_context)}: expected 256 records, got {len(records)}")
    payload = {
        "protocol": "strict_hepg2_lora_pool_init_ref_factorial_v1",
        "method": "ToDNA", "seed": seed, "target": "HepG2", "sequence_length": 200,
        "candidate_pool_context": pool_context, "logit_initialization": init_context,
        "kl_reference_context": reference_context, "kl_weight": kl_weight,
        "factorial_factors": {"pool_generation": pool_context, "logit_initialization": init_context, "kl_reference": reference_context},
        "pool": str(pool_path), "pool_sha256": sha256(pool_path),
        "starts": str(starts_path), "starts_sha256": sha256(starts_path),
        "logit_cache": str(cache_path), "logit_cache_sha256": sha256(cache_path),
        "design_checkpoint": str(design_checkpoint), "design_checkpoint_sha256": sha256(design_checkpoint),
        "adapter": str(adapter), "adapter_sha256": sha256(adapter),
        "config": {**CONFIG, "trust_region_weight": kl_weight,
                   "trajectory_noise_std": 0.02,
                   "trajectory_seed_base": seed * 100000,
                   "name": cell_name(pool_context, init_context, reference_context)},
        "pool_size": 768, "starts_per_pool": 128, "trajectories_per_start": 12,
        "steps": 10, "matched_optimization_queries": 15360,
        "total_design_reward_queries": 18432, "top_k": 256,
        "evaluation_model_used": False, "sealed_test_opened": False,
        "evaluation_boundary": "Evaluation and FullMPRA excluded until all factorial locks are complete",
        "search_summary": designer.last_search_summary,
        "cost_audit": profiler.summary(),
        "candidates": [record.to_dict() for record in records],
    }
    cell_root = output_root / cell_name(pool_context, init_context, reference_context) / f"seed{seed}"
    lock = cell_root / "formal_lock" / f"hepg2_seed{seed}_top256.json"
    write_json(lock, payload)
    write_json(cell_root / "formal_lock" / "manifest.json", {key: value for key, value in payload.items() if key != "candidates"})
    (cell_root / "formal_lock" / "complete").touch()
    (cell_root / "formal_search.complete").touch()
    write_json(cell_root / "progress.json", {"stage": "formal_search_complete", "seed": seed, "factorial_factors": payload["factorial_factors"]})
    return {"cell": cell_name(pool_context, init_context, reference_context), "lock": str(lock), "cost_audit": payload["cost_audit"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--lora-pool", type=Path, required=True)
    parser.add_argument("--base-starts", type=Path, required=True)
    parser.add_argument("--lora-starts", type=Path, required=True)
    parser.add_argument("--base-logit-cache", type=Path, required=True)
    parser.add_argument("--lora-logit-cache", type=Path, required=True)
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init-context", choices=LEVELS, required=True)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    args = parser.parse_args()
    seed_root = args.output_root / "group" / f"seed{args.seed}" / args.init_context
    if seed_root.exists():
        raise FileExistsError(f"refusing to overwrite {seed_root}")
    group_profiler = CostProfiler(method="ToDNA", protocol="strict_hepg2_lora_pool_init_ref_factorial_v1")
    with group_profiler.phase("model_load_and_design_predictor"):
        designer = FactorialDesigner(
            model_name="evo2_7b", lora_adapter=args.adapter if args.init_context == "lora" else None,
        )
        designer.dto.disable_flash_attention()
        predictor = load_activity_checkpoint(str(args.design_checkpoint), device=designer.dto.device)
    model_load_cost_audit = group_profiler.summary()
    results = []
    # Same loaded policy, four combinations of candidate pool and KL reference.
    for pool_context in LEVELS:
        for reference_context in LEVELS:
            cell = cell_name(pool_context, args.init_context, reference_context)
            complete_marker = (
                args.output_root / "search" / cell / f"seed{args.seed}" /
                "formal_search.complete"
            )
            if complete_marker.is_file():
                print(f"SKIP_COMPLETE_CELL {cell} seed={args.seed}", flush=True)
                results.append({"cell": cell, "skipped": True, "complete_marker": str(complete_marker)})
                continue
            results.append(run_cell(
                designer=designer, predictor=predictor,
                pool_path=(args.base_pool if pool_context == "base" else args.lora_pool),
                starts_path=(args.base_starts if pool_context == "base" else args.lora_starts),
                cache_path=(args.base_logit_cache if pool_context == "base" else args.lora_logit_cache),
                design_checkpoint=args.design_checkpoint, adapter=args.adapter,
                output_root=args.output_root / "search", seed=args.seed,
                pool_context=pool_context, init_context=args.init_context,
                reference_context=reference_context, kl_weight=args.kl_weight,
            ))
    group_root = args.output_root / "group" / f"seed{args.seed}" / args.init_context
    group_root.mkdir(parents=True, exist_ok=True)
    write_json(group_root / "manifest.json", {
        "protocol": "strict_hepg2_lora_pool_init_ref_factorial_v1",
        "seed": args.seed, "init_context": args.init_context,
        "model_load_cost_audit": model_load_cost_audit,
        "cells": results,
    })
    (group_root / "complete").touch()
    print(json.dumps({"seed": args.seed, "init_context": args.init_context, "cells": len(results), "group": str(group_root)}, indent=2))


if __name__ == "__main__":
    main()
