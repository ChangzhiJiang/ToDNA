#!/usr/bin/env python3
"""Run one cell of the Base/LoRA pool-init-reference 2x2x2 factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

from evo2_nabla_dna import CandidateRecord, Evo2TestTimeDesigner, SearchConfig, load_activity_checkpoint
from evo2_nabla_dna.dto import DtoConfig, RewardFn, validate_sequence
from evo2_nabla_dna.profiling import CostProfiler
from run_evo2_hparam_matched_seed import CtrlStyleDeltaR, load_pool, load_starts


CONFIG = {
    "steps": 10,
    "learning_rate": 0.1,
    "temperature": 0.7,
    "reward_weight": 1.0,
    "prior_weight": 0.0,
    "gradient_clip_norm": 5.0,
    "initial_margin": 0.25,
    "trust_region_weight": 1.0,
    "edit_penalty_weight": 0.0,
    "max_edits": None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sequence_metrics(sequence: str, evo2_nll: float) -> dict[str, float]:
    sequence = validate_sequence(sequence)
    gc_fraction = sum(base in "GC" for base in sequence) / len(sequence)
    best = current = 1
    for left, right in zip(sequence, sequence[1:]):
        current = current + 1 if left == right else 1
        best = max(best, current)
    return {"gc_fraction": float(gc_fraction), "max_homopolymer": float(best), "evo2_nll": float(evo2_nll)}


def load_cache(path: Path, starts: Sequence[str]) -> dict[str, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cached = payload.get("sequences")
    if cached != list(starts):
        raise ValueError("logit cache sequence ordering does not match starts")
    logits = payload.get("logits")
    if not isinstance(logits, dict) or set(logits) != {"base", "lora"}:
        raise ValueError("logit cache must contain Base and LoRA contexts")
    result: dict[str, dict[str, torch.Tensor]] = {"base": {}, "lora": {}}
    for context in ("base", "lora"):
        tensor = logits[context]
        if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != len(starts) or tensor.shape[1:] != (len(starts[0]), 4):
            raise ValueError(f"invalid {context} logit cache shape: {getattr(tensor, 'shape', None)}")
        result[context] = {sequence: tensor[index : index + 1].float() for index, sequence in enumerate(starts)}
    return result


class FactorialDesigner(Evo2TestTimeDesigner):
    def search_with_contexts(
        self,
        reward_fn: RewardFn,
        config: SearchConfig,
        trajectories: Sequence[str],
        contexts: dict[str, dict[str, torch.Tensor]],
        init_context: str,
        reference_context: str,
        profiler: CostProfiler,
    ) -> list[CandidateRecord]:
        records: list[CandidateRecord] = []
        seen: set[str] = set()
        duplicates = 0
        peak_memory = 0
        started_total = time.perf_counter()
        device = torch.device(self.dto.device)
        with profiler.phase(
            "optimization",
            trajectories=len(trajectories),
            matched_optimization_queries=len(trajectories) * config.dto.steps,
            design_reward_queries=len(trajectories) * (config.dto.steps + 2),
        ):
            for index, initial_sequence in enumerate(trajectories):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                dto_config = config.dto
                if dto_config.initial_logit_noise_std:
                    if dto_config.initial_logit_noise_seed is None:
                        raise ValueError("factorial trajectory noise requires a seed base")
                    dto_config = replace(
                        dto_config,
                        initial_logit_noise_seed=(
                            int(dto_config.initial_logit_noise_seed) + index
                        ) % (2**63 - 1),
                    )
                result = self.dto.optimize(
                    initial_sequence,
                    reward_fn,
                    dto_config,
                    initial_logits=contexts[init_context][initial_sequence],
                    reference_logits=(
                        contexts[reference_context][initial_sequence]
                        if config.dto.trust_region_weight != 0
                        else contexts[init_context][initial_sequence]
                    ),
                )
                if device.type == "cuda":
                    peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
                optimized = validate_sequence(result.optimized_sequence)
                if optimized in seen:
                    duplicates += 1
                    continue
                seen.add(optimized)
                log_probability = self.dto.score_sequence(optimized)
                records.append(
                    CandidateRecord(
                        initial_sequence=initial_sequence,
                        optimized_sequence=optimized,
                        initial_reward=result.initial_reward,
                        optimized_reward=result.final_reward,
                        evo2_log_probability=log_probability,
                        evo2_nll=-log_probability,
                        ranking_score=result.final_reward,
                        accepted=True,
                        constraint_metrics=sequence_metrics(optimized, -log_probability),
                        hamming_edits=result.hamming_edits,
                        trust_kl=result.final_trust_kl,
                        expected_edit_fraction=result.final_expected_edit_fraction,
                        history=result.history,
                        design_oracle_calls=config.dto.steps,
                        wall_clock_seconds=float(time.perf_counter() - started),
                        peak_gpu_memory_bytes=peak_memory,
                    )
                )
                if (index + 1) % 32 == 0 or index + 1 == len(trajectories):
                    print(f"FACTORIAL_PROGRESS attempted={index+1}/{len(trajectories)} unique={len(records)}", flush=True)

        records.sort(key=lambda record: record.ranking_score, reverse=True)
        self.last_search_summary = {
            "attempted": len(trajectories),
            "unique_projected": len(records),
            "accepted": len(records),
            "duplicates": duplicates,
            "optimization_stage_oracle_calls": len(trajectories) * config.dto.steps,
            "total_design_reward_calls": len(trajectories) * (config.dto.steps + 2),
            "post_optimization_discrete_score_calls": len(records),
            "total_wall_clock_seconds": time.perf_counter() - started_total,
            "peak_gpu_memory_bytes": peak_memory,
        }
        profiler.count("post_optimization_discrete_score_calls", len(records))
        profiler.count("duplicate_optimized_sequences", duplicates)
        return records[: config.top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--logit-cache", type=Path, required=True)
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool-context", choices=("base", "lora"), required=True)
    parser.add_argument("--init-context", choices=("base", "lora"), required=True)
    parser.add_argument("--reference-context", choices=("base", "lora"), required=True)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--trajectories-per-start", type=int, default=12)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if (args.top_k, args.trajectories_per_start) != (256, 12):
        raise ValueError("strict factorial requires Top-256 and 12 trajectories/start")

    pool = load_pool(args.pool)
    starts = load_starts(args.starts, set(pool))
    trajectories = [sequence for sequence in starts for _ in range(args.trajectories_per_start)]
    if len(pool) != 768 or len(starts) != 128 or len(trajectories) != 1536:
        raise ValueError("factorial pool/start invariant failed")
    contexts = load_cache(args.logit_cache, starts)
    profiler = CostProfiler(method="ToDNA", protocol="strict_hepg2_lora_pool_init_ref_factorial_v1")
    config_values = {**CONFIG, "trust_region_weight": args.kl_weight}
    config_values["trajectory_noise_std"] = 0.02
    config_values["trajectory_seed_base"] = args.seed * 100000
    config_values["name"] = f"pool_{args.pool_context}_init_{args.init_context}_ref_{args.reference_context}"
    search_config = SearchConfig(
        num_initial=len(trajectories),
        top_k=args.top_k,
        dto=DtoConfig(
            steps=10,
            learning_rate=0.1,
            temperature=0.7,
            reward_weight=1.0,
            prior_weight=0.0,
            gradient_clip_norm=5.0,
            initial_sequence_margin=0.25,
            trust_region_weight=args.kl_weight,
            edit_penalty_weight=0.0,
            max_edits=None,
            initial_logit_noise_std=0.02,
            initial_logit_noise_seed=args.seed * 100000,
        ),
        constraints=None,
        prior_penalty=0.0,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with profiler.phase("model_load_and_design_predictor"):
        designer = FactorialDesigner(model_name="evo2_7b", lora_adapter=args.adapter if args.init_context == "lora" else None)
        designer.dto.disable_flash_attention()
        predictor = load_activity_checkpoint(str(args.design_checkpoint), device=designer.dto.device)
        reward = CtrlStyleDeltaR(predictor, "hepg2").to(designer.dto.device)
    records = designer.search_with_contexts(
        reward_fn=reward,
        config=search_config,
        trajectories=trajectories,
        contexts=contexts,
        init_context=args.init_context,
        reference_context=args.reference_context,
        profiler=profiler,
    )
    if len(records) != args.top_k:
        raise RuntimeError(f"expected {args.top_k} locked candidates, got {len(records)}")

    payload = {
        "protocol": "strict_hepg2_lora_pool_init_ref_factorial_v1",
        "method": "ToDNA",
        "seed": args.seed,
        "target": "HepG2",
        "sequence_length": 200,
        "candidate_pool_context": args.pool_context,
        "logit_initialization": args.init_context,
        "kl_reference_context": args.reference_context,
        "kl_weight": args.kl_weight,
        "factorial_factors": {
            "pool_generation": args.pool_context,
            "logit_initialization": args.init_context,
            "kl_reference": args.reference_context,
        },
        "pool": str(args.pool),
        "pool_sha256": sha256(args.pool),
        "starts": str(args.starts),
        "starts_sha256": sha256(args.starts),
        "logit_cache": str(args.logit_cache),
        "logit_cache_sha256": sha256(args.logit_cache),
        "design_checkpoint": str(args.design_checkpoint),
        "design_checkpoint_sha256": sha256(args.design_checkpoint),
        "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter),
        "config": config_values,
        "config_sha256": json_hash(config_values),
        "pool_size": 768,
        "starts_per_pool": 128,
        "trajectories_per_start": 12,
        "steps": 10,
        "matched_optimization_queries": 15360,
        "total_design_reward_queries": 18432,
        "top_k": args.top_k,
        "evaluation_model_used": False,
        "sealed_test_opened": False,
        "evaluation_boundary": "Evaluation and FullMPRA excluded until all factorial locks are complete",
        "search_summary": designer.last_search_summary,
        "cost_audit": profiler.summary(),
        "candidates": [record.to_dict() for record in records],
    }
    lock = args.output_root / "formal_lock" / f"hepg2_seed{args.seed}_top256.json"
    write_json(lock, payload)
    write_json(args.output_root / "formal_lock" / "manifest.json", {key: value for key, value in payload.items() if key != "candidates"})
    (args.output_root / "formal_lock" / "complete").touch()
    (args.output_root / "formal_search.complete").touch()
    write_json(args.output_root / "progress.json", {"stage": "formal_search_complete", "seed": args.seed, "factorial_factors": payload["factorial_factors"]})
    print(json.dumps({"output": str(lock), "cost_audit": payload["cost_audit"]}, indent=2))


if __name__ == "__main__":
    main()
