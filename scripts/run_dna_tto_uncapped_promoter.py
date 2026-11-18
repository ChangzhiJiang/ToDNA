#!/usr/bin/env python3
"""Run the locked Evo2-TTO winner on immutable shared promoter starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from src import Evo2TestTimeDesigner
from promoter_matched_common import (
    RANGES, TASKS, load_pool, load_promoter_checkpoint, load_starts, padded_scores,
    sha256, write_json,
)
from run_search_sweep import build_search_config
from dna_tto_no_filter import NoSequenceConstraintDesigner


CONFIG = {
    "name": "hparam_winner_steps10_lr01",
    "steps": 10, "learning_rate": 0.1, "temperature": 0.7,
    "reward_weight": 1.0, "prior_weight": 0.5,
    "gradient_clip_norm": 5.0, "initial_margin": 0.25,
    "trust_region_weight": 1.0, "edit_penalty_weight": 2.0,
    "max_edits": None, "min_gc": 0.30, "max_gc": 0.70,
    "max_homopolymer": 6, "ranking_prior_penalty": 0.5,
}


def json_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


class PromoterDeltaR(nn.Module):
    def __init__(self, predictor: nn.Module, target_index: int):
        super().__init__()
        self.predictor = predictor
        self.target_index = target_index
        self.register_buffer("low", torch.tensor([RANGES[name][0] for name in TASKS]))
        self.register_buffer("high", torch.tensor([RANGES[name][1] for name in TASKS]))

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        normalized = (padded_scores(self.predictor, q) - self.low) / (self.high - self.low)
        off = [index for index in range(3) if index != self.target_index]
        return normalized[:, self.target_index] - normalized[:, off].mean(dim=1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()


def completed(path: Path, top_k: int) -> bool:
    return path.is_file() and len(json.loads(path.read_text(encoding="utf-8")).get("candidates", [])) == top_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--enformer-state-dict", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--shared-starts-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--trajectories-per-start", type=int, default=12)
    parser.add_argument("--edit-penalty-weight", type=float, default=2.0)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--ranking-prior-penalty", type=float, default=0.5)
    parser.add_argument("--disable-sequence-filters", action="store_true")
    args = parser.parse_args()

    config = {
        **CONFIG,
        "edit_penalty_weight": args.edit_penalty_weight,
        "prior_weight": args.prior_weight,
        "ranking_prior_penalty": args.ranking_prior_penalty,
    }
    if args.disable_sequence_filters:
        config["name"] = "dna_tto_uncapped_six_task_v1"
        config["enforce_hard_constraints"] = False
    if args.edit_penalty_weight == 0.0 and args.prior_weight == 0.0:
        config["name"] = "hparam_winner_steps10_lr01_prior0_edit0_candidate"
    elif args.edit_penalty_weight == 0.0:
        config["name"] = "hparam_winner_steps10_lr01_edit0_candidate"
    protocol = (
        "dna_tto_uncapped_six_task_v1"
        if args.disable_sequence_filters
        else
        "evo2_hparam_winner_public_promoter_strict_top256_v1"
        if args.edit_penalty_weight == 2.0 and args.prior_weight == 0.5
        else "evo2_tto_prior0_candidate_public_promoter_strict_top256_v1"
        if args.prior_weight == 0.0
        else "evo2_tto_edit0_candidate_public_promoter_strict_top256_v1"
    )

    config["name"] = protocol = "dna_tto_uncapped_six_task_v1"
    assert args.disable_sequence_filters and config["max_edits"] is None
    assert config["prior_weight"] == config["edit_penalty_weight"] == config["ranking_prior_penalty"] == 0.0
    calls = 128 * args.trajectories_per_start * int(config["steps"])
    if calls != 15360:
        raise ValueError(f"strict budget requires 15360 calls, got {calls}")
    if not (args.shared_starts_root / "complete").exists():
        raise FileNotFoundError("shared promoter Top-128 is not complete")
    pool = load_pool(args.pool)
    complete = args.output_root / "formal_search.complete"
    if complete.exists():
        print(f"EVO2_PROMOTER_STRICT_ALREADY_COMPLETE seed={args.seed}", flush=True)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    designer_class = NoSequenceConstraintDesigner if args.disable_sequence_filters else Evo2TestTimeDesigner
    designer = designer_class(model_name="evo2_7b", lora_adapter=args.adapter)
    designer.dto.disable_flash_attention()
    predictor, task_order = load_promoter_checkpoint(
        args.design_checkpoint, args.enformer_state_dict, designer.dto.device
    )
    lock_dir = args.output_root / "formal_lock"
    files = []
    start_hashes = {}
    for target_index, target in enumerate(TASKS):
        lock = lock_dir / f"{target.lower()}_seed{args.seed}_top{args.top_k}.json"
        files.append(str(lock))
        start_path = args.shared_starts_root / f"{target.lower()}_starting_top128.json"
        starts = load_starts(start_path, set(pool))
        start_hashes[target] = sha256(start_path)
        if completed(lock, args.top_k):
            continue
        trajectories = [value for value in starts for _ in range(args.trajectories_per_start)]
        target_seed = (args.seed * 1000 + target_index * 10 + 7) % (2**32)
        random.seed(target_seed)
        torch.manual_seed(target_seed)
        write_json(args.output_root / "progress.json", {
            "stage": "evo2_search", "seed": args.seed, "target": target,
            "trajectories": len(trajectories), "steps": config["steps"],
            "oracle_sequence_evaluations": calls,
        })
        reward = PromoterDeltaR(predictor, target_index).to(designer.dto.device)
        search_config = build_search_config(config, len(trajectories), args.top_k)
        if args.disable_sequence_filters:
            search_config = replace(search_config, constraints=None)
        records = designer.search(
            prompt="ACGT", length=250, reward_fn=reward,
            config=search_config,
            initial_sequences=trajectories,
        )
        if len(records) != args.top_k:
            raise RuntimeError(f"{target}: expected Top-{args.top_k}, got {len(records)}")
        write_json(lock, {
            "protocol": protocol,
            "seed": args.seed, "search_seed": target_seed, "target": target,
            "tasks": list(task_order), "sequence_length": 250,
            "shared_pool": str(args.pool), "pool_sha256": sha256(args.pool),
            "shared_top128": str(start_path), "shared_top128_sha256": sha256(start_path),
            "design_checkpoint": str(args.design_checkpoint),
            "design_checkpoint_sha256": sha256(args.design_checkpoint),
            "adapter": str(args.adapter), "adapter_sha256": sha256(args.adapter),
            "config": config, "config_sha256": json_hash(config),
            "trajectory_expansion": f"{args.trajectories_per_start} independent trajectories per shared Top-128 start",
            "oracle_sequence_evaluations_per_target": calls, "top_k": args.top_k,
            "activity_reward": "published-range normalized target minus arithmetic-mean off-target",
            "evaluation_boundary": "Evaluation and all external metrics excluded from search and locking",
            "hard_constraint_policy": "GC and homopolymer rejection disabled" if args.disable_sequence_filters else "configured GC/homopolymer rejection enabled",
            "search_summary": designer.last_search_summary,
            "candidates": [record.to_dict() for record in records],
        })
        (lock_dir / f"{target.lower()}.complete").touch()
        print(f"EVO2_PROMOTER_STRICT_TARGET_COMPLETE target={target}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if all(completed(Path(path), args.top_k) for path in files):
        write_json(lock_dir / "manifest.json", {
            "protocol": protocol,
            "seed": args.seed, "targets": list(TASKS), "candidate_files": files,
            "pool_sha256": sha256(args.pool), "shared_top128_sha256": start_hashes,
            "design_checkpoint_sha256": sha256(args.design_checkpoint),
            "adapter_sha256": sha256(args.adapter), "config_sha256": json_hash(config),
            "top_k": args.top_k, "oracle_evaluations_per_target": calls,
        })
        (lock_dir / "complete").touch()
        complete.touch()
        write_json(args.output_root / "progress.json", {"stage": "formal_search_complete", "seed": args.seed})
        print(f"EVO2_PROMOTER_STRICT_SEED_COMPLETE seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
