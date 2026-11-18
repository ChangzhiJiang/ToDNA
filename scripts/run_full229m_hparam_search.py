#!/usr/bin/env python3
"""Two-stage Design-Model-only hyperparameter search for Evo 2 DTO."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from src import Evo2TestTimeDesigner, MultiCellActivityReward, load_activity_checkpoint
from run_full229m_formal import generate_pool, load_pool, sha256
from run_search_sweep import build_search_config, candidate_summary, load_configs


CELL_NAMES = ("K562", "HepG2", "SK-N-SH")
STANDARD_PRIOR_PENALTY = 0.5
MIN_ACCEPTANCE_RATE = 0.45


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def target_metrics(records: list[object], search_summary: dict[str, object]) -> dict[str, float]:
    attempted = int(search_summary["attempted"])
    accepted = int(search_summary["accepted"])
    acceptance_rate = accepted / attempted
    if not records:
        return {
            "objective": -1e9,
            "accepted": float(accepted),
            "acceptance_rate": acceptance_rate,
            "mean_design_reward": -1e9,
            "mean_design_reward_change": -1e9,
            "mean_standardized_ranking": -1e9,
            "mean_evo2_nll": 1e9,
            "mean_hamming_edits": 200.0,
            "positional_entropy_bits": 0.0,
        }
    summary = candidate_summary(records)
    design_reward = mean([float(record.optimized_reward) for record in records])
    standardized_ranking = mean(
        [
            float(record.optimized_reward)
            - STANDARD_PRIOR_PENALTY * float(record.evo2_nll)
            for record in records
        ]
    )
    edits = float(summary["mean_hamming_edits"])
    entropy = float(summary["mean_positional_entropy_bits"])
    acceptance_penalty = 5.0 * max(0.0, MIN_ACCEPTANCE_RATE - acceptance_rate)
    objective = (
        standardized_ranking
        + 0.5 * acceptance_rate
        + 0.1 * entropy
        - 0.002 * edits
        - acceptance_penalty
    )
    return {
        "objective": objective,
        "accepted": float(accepted),
        "acceptance_rate": acceptance_rate,
        "mean_design_reward": design_reward,
        "mean_design_reward_change": float(summary["mean_design_reward_change"]),
        "mean_standardized_ranking": standardized_ranking,
        "mean_evo2_nll": float(summary["mean_evo2_nll"]),
        "mean_hamming_edits": edits,
        "positional_entropy_bits": entropy,
    }


def aggregate_config(targets: dict[str, dict[str, float]]) -> dict[str, object]:
    target_scores = [targets[target]["objective"] for target in CELL_NAMES]
    mean_score = mean(target_scores)
    worst_score = min(target_scores)
    return {
        "objective": 0.75 * mean_score + 0.25 * worst_score,
        "mean_target_objective": mean_score,
        "worst_target_objective": worst_score,
        "targets": targets,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_stage(
    designer: Evo2TestTimeDesigner,
    predictor: torch.nn.Module,
    configs: list[dict[str, object]],
    sequences: list[str],
    pool_path: Path,
    seed: int,
    stage_name: str,
    output_root: Path,
    progress_path: Path,
) -> list[dict[str, object]]:
    random.seed(seed)
    torch.manual_seed(seed)
    by_config: dict[str, dict[str, dict[str, float]]] = {
        str(config["name"]): {} for config in configs
    }
    stage_dir = output_root / stage_name
    for target in CELL_NAMES:
        target_index = CELL_NAMES.index(target)
        reward = MultiCellActivityReward(
            predictor=predictor,
            target_index=target_index,
            off_target_indices=[index for index in range(3) if index != target_index],
            smooth_max_temperature=0.1,
        )
        for config_index, values in enumerate(configs, start=1):
            name = str(values["name"])
            write_json(
                progress_path,
                {
                    "stage": stage_name,
                    "seed": seed,
                    "target": target,
                    "config": name,
                    "config_index": config_index,
                    "config_count": len(configs),
                    "pool_size": len(sequences),
                },
            )
            records = designer.search(
                prompt="ACGT",
                length=200,
                reward_fn=reward,
                config=build_search_config(values, len(sequences), len(sequences)),
                initial_sequences=sequences,
            )
            metrics = target_metrics(records, designer.last_search_summary)
            by_config[name][target] = metrics
            payload = {
                "protocol": "v2_design_model_only_hparam_search",
                "stage": stage_name,
                "seed": seed,
                "target": target,
                "pool_file": str(pool_path),
                "pool_sha256": sha256(pool_path),
                "config": values,
                "search_summary": designer.last_search_summary,
                "selection_metrics": metrics,
                "candidates": [record.to_dict() for record in records],
            }
            slug = target.lower().replace("-", "")
            write_json(stage_dir / f"{slug}_{name}.json", payload)
            print(
                f"HPARAM_RESULT stage={stage_name} target={target} config={name} "
                f"objective={metrics['objective']:.6f} "
                f"accepted={int(metrics['accepted'])}/{len(sequences)}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ranking: list[dict[str, object]] = []
    by_name = {str(config["name"]): config for config in configs}
    for name, targets in by_config.items():
        aggregate = aggregate_config(targets)
        ranking.append({"name": name, "config": by_name[name], **aggregate})
    ranking.sort(key=lambda item: float(item["objective"]), reverse=True)
    write_json(stage_dir / "ranking.json", ranking)
    print(
        f"HPARAM_RANKING stage={stage_name} "
        + " ".join(
            f"{index + 1}:{item['name']}={float(item['objective']):.6f}"
            for index, item in enumerate(ranking)
        ),
        flush=True,
    )
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/full229m_hparam_v2")
    )
    args = parser.parse_args()

    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    configs = load_configs(args.config)
    stage1 = protocol["stage1"]
    stage2 = protocol["stage2"]
    progress_path = args.output_root / "progress.json"
    write_json(progress_path, {"stage": "loading_evo2"})
    designer = Evo2TestTimeDesigner(model_name="evo2_7b")
    designer.dto.disable_flash_attention()

    pools: dict[str, tuple[Path, list[str], int]] = {}
    all_sequences: set[str] = set()
    for stage_name, stage in (("stage1", stage1), ("stage2", stage2)):
        seed = int(stage["seed"])
        count = int(stage["pool_size"])
        pool_path = args.data_dir / f"hparam_v2_{stage_name}_seed{seed}_n{count}.json"
        sequences = load_pool(pool_path, seed, count, 200)
        if sequences is None:
            write_json(
                progress_path,
                {"stage": "generating_pool", "search_stage": stage_name, "seed": seed},
            )
            sequences = generate_pool(designer, pool_path, seed, count, 200)
        overlap = all_sequences.intersection(sequences)
        if overlap:
            raise ValueError(f"hyperparameter pools overlap: {len(overlap)} sequences")
        all_sequences.update(sequences)
        pools[stage_name] = (pool_path, sequences, seed)

    write_json(progress_path, {"stage": "loading_design_model"})
    predictor = load_activity_checkpoint(str(args.design_checkpoint), device=designer.dto.device)

    stage1_path, stage1_sequences, stage1_seed = pools["stage1"]
    ranking1 = run_stage(
        designer,
        predictor,
        configs,
        stage1_sequences,
        stage1_path,
        stage1_seed,
        "stage1",
        args.output_root,
        progress_path,
    )
    keep = int(stage1["keep"])
    finalists = [item["config"] for item in ranking1[:keep]]
    stage2_path, stage2_sequences, stage2_seed = pools["stage2"]
    ranking2 = run_stage(
        designer,
        predictor,
        finalists,
        stage2_sequences,
        stage2_path,
        stage2_seed,
        "stage2",
        args.output_root,
        progress_path,
    )
    winner = ranking2[0]
    summary = {
        "protocol": protocol["protocol"],
        "selection_objective": protocol["selection_objective"],
        "design_checkpoint": str(args.design_checkpoint),
        "design_checkpoint_sha256": sha256(args.design_checkpoint),
        "stage1_pool": str(stage1_path),
        "stage1_ranking": ranking1,
        "stage2_pool": str(stage2_path),
        "stage2_ranking": ranking2,
        "winner": winner,
        "evaluation_boundary": "Evaluation-Model and FullMPRA were not loaded or queried",
    }
    write_json(args.output_root / "summary.json", summary)
    (args.output_root / "complete").touch()
    write_json(
        progress_path,
        {"stage": "complete", "winner": winner["name"], "objective": winner["objective"]},
    )
    print(
        f"HPARAM_SEARCH_COMPLETE winner={winner['name']} "
        f"objective={float(winner['objective']):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
