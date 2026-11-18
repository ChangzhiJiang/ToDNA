#!/usr/bin/env python3
"""DNA_TTO: constraint-free candidate selection with Table 3 component ablations."""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import pickle
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch
import evo2

try:
    import numpy as np
except ImportError:  # pragma: no cover - the production environment includes NumPy
    np = None

from src import (
    CandidateRecord,
    Evo2TestTimeDesigner,
    SearchConfig,
    load_activity_checkpoint,
)
from src.dto import RewardFn, validate_sequence
from run_evo2_hparam_matched_seed import CtrlStyleDeltaR, load_pool, load_starts
from run_search_sweep import build_search_config


_ORIGINAL_EVO2 = evo2.Evo2


def _offline_evo2(model_name: str, *args, **kwargs):
    local_path = os.environ.get("EVO2_LOCAL_PATH")
    if local_path:
        kwargs["local_path"] = local_path
    return _ORIGINAL_EVO2(model_name, *args, **kwargs)


evo2.Evo2 = _offline_evo2


CONFIG = {
    "name": "DNA_TTO",
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
    "min_gc": 0.30,
    "max_gc": 0.70,
    "max_homopolymer": 6,
    "ranking_prior_penalty": 0.0,
    "enforce_hard_constraints": False,
    # Opt-in for the repaired replay protocol.  Historical locks retain the
    # zero-noise setting; new cost/replay-matched runs pass a positive value.
    "trajectory_noise_std": 0.0,
    "trajectory_seed_base": None,
    "audit_hashes": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def bytes_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rng_snapshot() -> dict[str, object]:
    """Hash process RNG states without serializing sensitive tensor contents."""
    snapshot: dict[str, object] = {
        "python_random_sha256": bytes_hash(pickle.dumps(random.getstate(), protocol=4)),
        "torch_cpu_sha256": bytes_hash(torch.get_rng_state().cpu().numpy().tobytes()),
    }
    if np is not None:
        snapshot["numpy_sha256"] = bytes_hash(pickle.dumps(np.random.get_state(), protocol=4))
    else:
        snapshot["numpy_sha256"] = None
    if torch.cuda.is_available():
        snapshot["torch_cuda_sha256"] = [
            bytes_hash(state.cpu().numpy().tobytes())
            for state in torch.cuda.get_rng_state_all()
        ]
    else:
        snapshot["torch_cuda_sha256"] = []
    return snapshot


def seed_audit_trajectory(seed: int) -> None:
    """Reset all process RNGs at a trajectory boundary for replay audits."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed % (2**63 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed % (2**63 - 1))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sequence_metrics(sequence: str, evo2_nll: float) -> dict[str, float]:
    sequence = validate_sequence(sequence)
    gc_fraction = sum(base in "GC" for base in sequence) / len(sequence)
    max_run = 1
    current_run = 1
    for previous, current in zip(sequence, sequence[1:]):
        current_run = current_run + 1 if current == previous else 1
        max_run = max(max_run, current_run)
    return {
        "gc_fraction": float(gc_fraction),
        "max_homopolymer": float(max_run),
        "evo2_nll": float(evo2_nll),
    }


class DNATTODesigner(Evo2TestTimeDesigner):
    """Optimize and deduplicate candidates without GC/homopolymer rejection."""

    def search(
        self,
        prompt: str,
        length: int,
        reward_fn: RewardFn,
        config: SearchConfig | None = None,
        initial_sequences: Sequence[str] | None = None,
    ) -> list[CandidateRecord]:
        config = config or SearchConfig()
        if config.constraints is not None:
            raise ValueError("sequence-constraint ablation requires constraints=None")
        if initial_sequences is None:
            raise ValueError("formal ablation requires locked initial sequences")
        sequences = [validate_sequence(sequence) for sequence in initial_sequences]
        if not sequences or any(len(sequence) != length for sequence in sequences):
            raise ValueError("all initial sequences must match the requested length")

        records: list[CandidateRecord] = []
        seen: set[str] = set()
        duplicates = 0
        self.trajectory_audit: list[dict[str, object]] = []
        started_total = time.perf_counter()
        device = torch.device(self.dto.device)
        peak_memory = 0
        for candidate_index, initial_sequence in enumerate(sequences, start=1):
            dto_config = config.dto
            if dto_config.initial_logit_noise_std:
                base_seed = dto_config.initial_logit_noise_seed
                if base_seed is None:
                    raise ValueError("trajectory seed base is required for logit noise")
                dto_config = replace(
                    dto_config,
                    initial_logit_noise_seed=(base_seed + candidate_index - 1)
                    % (2**63 - 1),
                )
            started = time.perf_counter()
            if config.dto.audit_hashes:
                audit_seed = int(dto_config.initial_logit_noise_seed or candidate_index)
                seed_audit_trajectory(audit_seed)
            rng_before = rng_snapshot() if config.dto.audit_hashes else None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            result = self.dto.optimize(initial_sequence, reward_fn, dto_config)
            rng_after = rng_snapshot() if config.dto.audit_hashes else None
            elapsed = time.perf_counter() - started
            candidate_peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            )
            peak_memory = max(peak_memory, candidate_peak_memory)
            if candidate_index == 8 and getattr(self, "memory_probe_path", None):
                write_json(self.memory_probe_path, {
                    "completed_trajectories": candidate_index,
                    "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "free_mib": torch.cuda.mem_get_info()[0] / 2**20,
                })
                print("MEMORY_PROBE_PASSED trajectories=8", flush=True)
            optimized_sequence = validate_sequence(result.optimized_sequence)
            duplicate = optimized_sequence in seen
            if config.dto.audit_hashes:
                self.trajectory_audit.append({
                    "trajectory_index": candidate_index - 1,
                    "initial_sequence": initial_sequence,
                    "optimized_sequence": optimized_sequence,
                    "duplicate": duplicate,
                    "initial_logit_noise_seed": dto_config.initial_logit_noise_seed,
                    "initial_noise_sha256": result.initial_noise_sha256,
                    "initial_logits_sha256": result.initial_logits_sha256,
                    "final_logits_sha256": result.final_logits_sha256,
                    "rng_before": rng_before,
                    "rng_after": rng_after,
                })
            if duplicate:
                duplicates += 1
                continue
            seen.add(optimized_sequence)
            log_probability = self.dto.score_sequence(optimized_sequence)
            evo2_nll = -log_probability
            records.append(
                CandidateRecord(
                    initial_sequence=initial_sequence,
                    optimized_sequence=optimized_sequence,
                    initial_reward=result.initial_reward,
                    optimized_reward=result.final_reward,
                    evo2_log_probability=log_probability,
                    evo2_nll=evo2_nll,
                    ranking_score=result.final_reward,
                    accepted=True,
                    constraint_metrics=sequence_metrics(optimized_sequence, evo2_nll),
                    hamming_edits=result.hamming_edits,
                    trust_kl=result.final_trust_kl,
                    expected_edit_fraction=result.final_expected_edit_fraction,
                    history=result.history,
                    design_oracle_calls=int(dto_config.steps),
                    wall_clock_seconds=float(elapsed),
                    peak_gpu_memory_bytes=candidate_peak_memory,
                )
            )
            if candidate_index % 32 == 0 or candidate_index == len(sequences):
                print(
                    f"SEARCH_PROGRESS attempted={candidate_index}/{len(sequences)} "
                    f"unique={len(records)} accepted={len(records)}",
                    flush=True,
                )

        records.sort(key=lambda record: record.ranking_score, reverse=True)
        self.last_search_summary = {
            "attempted": len(sequences),
            "unique_decoded": len(records),
            "unique_projected": len(records),
            "accepted": len(records),
            "duplicates": duplicates,
            "rejection_counts": {},
            "hard_constraints_enabled": False,
            "total_wall_clock_seconds": float(time.perf_counter() - started_total),
            "mean_trajectory_wall_clock_seconds": float(
                statistics.fmean(record.wall_clock_seconds for record in records)
            ) if records else 0.0,
            "peak_gpu_memory_bytes": int(peak_memory),
            "optimization_stage_oracle_calls": int(len(sequences) * config.dto.steps),
            "post_optimization_discrete_score_calls": int(len(records)),
        }
        return records[: config.top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("full", "no_trust", "no_lora"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--pool-context",
        choices=("base", "lora"),
        default="lora",
        help="generator that produced the immutable candidate pool",
    )
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--trajectories-per-start", type=int, default=12)
    parser.add_argument(
        "--trajectory-noise-std",
        type=float,
        default=0.0,
        help="CPU Gaussian initial-logit noise for the replay protocol; 0 preserves historical runs",
    )
    parser.add_argument(
        "--audit-hashes",
        action="store_true",
        help="record per-trajectory RNG and logit hashes for deterministic replay audits",
    )
    args = parser.parse_args()
    config = dict(CONFIG)
    if args.variant == "no_trust":
        config["trust_region_weight"] = 0.0
    config["name"] = "DNA_TTO" if args.variant == "full" else "DNA_TTO_" + args.variant
    if args.trajectory_noise_std < 0:
        raise ValueError("trajectory-noise-std must be non-negative")
    config["trajectory_noise_std"] = args.trajectory_noise_std
    config["trajectory_seed_base"] = (
        (args.seed * 1_000_000 + 17) % (2**63 - 1)
        if args.trajectory_noise_std else None
    )
    config["audit_hashes"] = bool(args.audit_hashes)
    frozen = args.variant == "no_lora"
    if frozen:
        # The queue may pass the shared adapter argument to every variant.
        # no_lora must still instantiate the base Evo2 model only.
        args.adapter = None
    elif args.adapter is None:
        raise ValueError("full and no_trust require a LoRA adapter")
    optimized = args.variant != "no_tto"

    if args.top_k != 256 or args.trajectories_per_start != 12:
        raise ValueError("strict protocol requires Top-256 and 12 trajectories/start")
    pool = load_pool(args.pool)
    starts = load_starts(args.starts, set(pool))
    trajectories = [sequence for sequence in starts for _ in range(12)] if optimized else pool
    calls = len(starts) * 12 * int(config["steps"])
    if len(pool) != 768 or len(starts) != 128 or calls != (15360 if optimized else 0):
        raise ValueError("strict pool/start/oracle-call invariant failed")

    assert config["max_edits"] is None and not config["enforce_hard_constraints"]
    if args.dry_run:
        print(json.dumps({"variant": args.variant, "config": config,
                          "pool_sha256": sha256(args.pool), "starts_sha256": sha256(args.starts),
                          "optimization_calls": calls, "attempts": len(trajectories),
                          "output": str(args.output_root)}, indent=2))
        return

    lock = args.output_root / "formal_lock" / f"hepg2_seed{args.seed}_top256.json"
    if lock.is_file():
        payload = json.loads(lock.read_text(encoding="utf-8"))
        if (len(payload.get("candidates", [])) == args.top_k
                and payload.get("config_sha256") == json_hash(config)
                and payload.get("pool_sha256") == sha256(args.pool)
                and payload.get("shared_top128_sha256") == sha256(args.starts)):
            if not (args.output_root / "formal_search.complete").is_file():
                raise RuntimeError("Existing lock has no completion marker; audit before resuming")
            print(f"DNA_TTO_ALREADY_COMPLETE seed={args.seed}")
            return
        raise RuntimeError(f"incomplete pre-existing lock must be audited: {lock}")

    search_config = replace(
        build_search_config(config, len(trajectories), args.top_k), constraints=None
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    search_seed = (args.seed * 1000 + 17) % (2**32)
    random.seed(search_seed)
    torch.manual_seed(search_seed)
    write_json(
        args.output_root / "progress.json",
        {
            "stage": "search",
            "variant": args.variant,
            "seed": args.seed,
            "initial_sequences": len(trajectories),
            "steps": config["steps"],
            "optimization_oracle_calls": calls,
        },
    )

    designer = DNATTODesigner(
        model_name="evo2_7b", lora_adapter=args.adapter
    )
    designer.dto.disable_flash_attention()
    designer.memory_probe_path = args.output_root / "memory_probe.json"
    predictor = load_activity_checkpoint(
        str(args.design_checkpoint), device=designer.dto.device
    )
    reward = CtrlStyleDeltaR(predictor, "hepg2").to(designer.dto.device)
    records = designer.search(
        prompt="ACGT",
        length=200,
        reward_fn=reward,
        config=search_config,
        initial_sequences=trajectories,
    )
    if len(records) != args.top_k:
        raise RuntimeError(f"expected Top-{args.top_k}, got {len(records)}")

    if not optimized and any(r.hamming_edits or r.initial_sequence != r.optimized_sequence for r in records):
        raise RuntimeError("Zero-step control must retain unchanged pool sequences")
    payload = {
        "protocol": "strict_hepg2_lora_pool_logit_factorial_v1",
        "model_name": "DNA_TTO",
        "comparison_role": ("independent_pool_context" if args.variant == "frozen_context" else
                            "strict_paired_ablation" if optimized else "structural_zero_optimization_control"),
        "model_context": "frozen" if frozen else "lora",
        "variant": args.variant,
        "seed": args.seed,
        "search_seed": search_seed,
        "target": "HepG2",
        "sequence_length": 200,
        "shared_pool": str(args.pool),
        "pool_sha256": sha256(args.pool),
        "shared_top128": str(args.starts),
        "shared_top128_sha256": sha256(args.starts),
        "top128_used_as_optimization_starts": optimized,
        "initial_sequence_count": len(trajectories),
        "trajectory_expansion": "12 attempts per Top-128 start" if optimized else "unchanged shared pool",
        "trajectory_perturbation": {
            "type": "cpu_gaussian_initial_logit_noise" if config["trajectory_noise_std"] else "none",
            "std": config["trajectory_noise_std"],
            "seed_base": config["trajectory_seed_base"],
            "seed_schedule": "seed_base + flattened trajectory index" if config["trajectory_seed_base"] is not None else None,
        },
        "optimization_oracle_evaluations": calls,
        "selection_oracle_evaluations": 0 if optimized else len(pool),
        "top_k": args.top_k,
        "design_objective": "published-range normalized HepG2 minus arithmetic-mean off-target",
        "config": config,
        "config_sha256": json_hash(config),
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": sha256(args.adapter) if args.adapter else None,
        "design_checkpoint": str(args.design_checkpoint),
        "design_checkpoint_sha256": sha256(args.design_checkpoint),
        "decode_policy": "unrestricted argmax; no edit cap",
        "initialization": "base Evo2 logits" if frozen else "LoRA Evo2 logits",
        "candidate_pool_context": args.pool_context,
        "factorial_factors": {
            "pool_generation": args.pool_context,
            "logit_initialization": "base" if frozen else "lora",
        },
        "hard_constraint_policy": "GC and homopolymer rejection disabled",
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "source_hashes": {str(p): sha256(p) for p in (
            Path(__file__), Path(__file__).resolve().parents[1] / "src/dto.py")},
        "evaluation_boundary": "Evaluation and FullMPRA excluded until the immutable Top-256 lock",
        "search_summary": dict(designer.last_search_summary),
        "candidates": [record.to_dict() for record in records],
    }
    if config["audit_hashes"]:
        audit_path = args.output_root / "trajectory_audit.jsonl"
        audit_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in designer.trajectory_audit),
            encoding="utf-8",
        )
        payload["trajectory_audit_path"] = str(audit_path)
        payload["trajectory_audit_sha256"] = sha256(audit_path)
    write_json(lock, payload)
    write_json(
        args.output_root / "formal_lock" / "manifest.json",
        {key: value for key, value in payload.items() if key != "candidates"},
    )
    (args.output_root / "formal_lock" / "complete").touch()
    (args.output_root / "formal_search.complete").touch()
    write_json(
        args.output_root / "progress.json",
        {"stage": "formal_search_complete", "variant": args.variant, "seed": args.seed},
    )
    print(f"DNA_TTO_SEARCH_COMPLETE seed={args.seed}")


if __name__ == "__main__":
    main()
