#!/usr/bin/env python3
"""Generate fixed Evo 2 pools and run the locked full-229M formal search."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from src import (
    Evo2TestTimeDesigner,
    MultiCellActivityReward,
    SearchConfig,
    load_activity_checkpoint,
)
from run_search_sweep import build_search_config, candidate_summary, load_configs


CELL_NAMES = ("K562", "HepG2", "SK-N-SH")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sequences(sequences: object, expected: int, length: int) -> list[str]:
    if not isinstance(sequences, list) or len(sequences) != expected:
        raise ValueError(f"expected exactly {expected} initial sequences")
    if not all(
        isinstance(sequence, str)
        and len(sequence) == length
        and not (set(sequence) - set("ACGT"))
        for sequence in sequences
    ):
        raise ValueError("initial pool contains invalid DNA")
    if len(set(sequences)) != expected:
        raise ValueError("initial pool contains duplicate sequences")
    return sequences


def load_pool(path: Path, seed: int, expected: int, length: int) -> list[str] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("seed") != seed:
        raise ValueError(f"pool metadata does not match seed {seed}: {path}")
    return validate_sequences(payload.get("sequences"), expected, length)


def generate_pool(
    designer: Evo2TestTimeDesigner,
    path: Path,
    seed: int,
    count: int,
    length: int,
) -> list[str]:
    random.seed(seed)
    torch.manual_seed(seed)
    sequences: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 4
    while len(sequences) < count and attempts < max_attempts:
        config = SearchConfig(
            num_initial=1,
            initial_temperature=1.0,
            initial_top_k=4,
            initial_top_p=1.0,
        )
        for sequence in designer.sample_initial("ACGT", length, config):
            if sequence not in seen:
                seen.add(sequence)
                sequences.append(sequence)
        attempts += 1
        if len(sequences) % 16 == 0 or len(sequences) == count:
            print(
                f"POOL seed={seed} unique={len(sequences)}/{count} attempts={attempts}",
                flush=True,
            )
    validate_sequences(sequences, count, length)
    payload = {
        "model": "evo2_7b",
        "seed": seed,
        "prompt": "ACGT",
        "length": length,
        "num_sequences": count,
        "generation_batch_size": 1,
        "generation_attempts": attempts,
        "temperature": 1.0,
        "top_k": 4,
        "top_p": 1.0,
        "sequences": sequences,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"POOL_COMPLETE seed={seed} sha256={sha256(path)} path={path}", flush=True)
    return sequences


def raw_output_is_complete(path: Path, top_k: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["candidates"]
        return (
            isinstance(records, list)
            and len(records) == top_k
            and all(record.get("accepted") is True for record in records)
            and all("evaluation" not in record for record in records)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_progress(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20240827, 20240828, 20240829]
    )
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()

    lock_payload = json.loads(args.lock.read_text(encoding="utf-8"))
    configs = load_configs(args.lock)
    if len(configs) != 1:
        raise ValueError("formal lock must contain exactly one configuration")
    values = configs[0]
    proposal_budget = int(lock_payload["proposal_budget"])
    top_k = int(lock_payload["top_k"])
    expected_hash = str(lock_payload["design_checkpoint_sha256"])
    actual_hash = sha256(args.design_checkpoint)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Design-Model hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "full229m_formal_progress.json"
    write_progress(progress_path, stage="loading_evo2", seeds=args.seeds)
    designer = Evo2TestTimeDesigner(model_name="evo2_7b")
    designer.dto.disable_flash_attention()

    pools: dict[int, tuple[Path, list[str]]] = {}
    all_sequences: set[str] = set()
    for seed in args.seeds:
        pool_path = args.data_dir / f"initial_pool_seed{seed}_n{proposal_budget}.json"
        sequences = load_pool(pool_path, seed, proposal_budget, args.length)
        if sequences is None:
            write_progress(progress_path, stage="generating_pool", seed=seed)
            sequences = generate_pool(
                designer, pool_path, seed, proposal_budget, args.length
            )
        overlap = all_sequences.intersection(sequences)
        if overlap:
            raise ValueError(f"formal pools overlap before seed {seed}: {len(overlap)}")
        all_sequences.update(sequences)
        pools[seed] = (pool_path, sequences)

    write_progress(progress_path, stage="loading_design_model")
    predictor = load_activity_checkpoint(
        str(args.design_checkpoint), device=designer.dto.device
    )
    config_name = str(values["name"])

    for seed in args.seeds:
        pool_path, sequences = pools[seed]
        random.seed(seed)
        torch.manual_seed(seed)
        output_dir = args.output_root / f"full229m_formal_seed{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for target in CELL_NAMES:
            target_slug = target.lower().replace("-", "")
            output = output_dir / f"{target_slug}_{config_name}_seed{seed}.json"
            if raw_output_is_complete(output, top_k):
                print(f"SKIP_COMPLETE seed={seed} target={target} path={output}", flush=True)
                continue
            write_progress(
                progress_path,
                stage="formal_search",
                seed=seed,
                target=target,
                proposal_budget=proposal_budget,
                top_k=top_k,
            )
            target_index = CELL_NAMES.index(target)
            reward = MultiCellActivityReward(
                predictor=predictor,
                target_index=target_index,
                off_target_indices=[i for i in range(3) if i != target_index],
                smooth_max_temperature=0.1,
            )
            records = designer.search(
                prompt="ACGT",
                length=args.length,
                reward_fn=reward,
                config=build_search_config(values, proposal_budget, top_k),
                initial_sequences=sequences,
            )
            if len(records) != top_k:
                raise RuntimeError(
                    f"{target} seed {seed} returned {len(records)} candidates; "
                    f"the locked protocol requires Top-{top_k}"
                )
            payload = {
                "method": "evo2_7b_nabla_test_time_design",
                "protocol": "locked_full229m_formal",
                "selection_boundary": "design_model_evo2_and_hard_constraints_only",
                "target": target,
                "cell_order": CELL_NAMES,
                "seed": seed,
                "length": args.length,
                "num_initial": proposal_budget,
                "top_k": top_k,
                "initial_sequences_file": str(pool_path),
                "initial_sequences_sha256": sha256(pool_path),
                "design_checkpoint": str(args.design_checkpoint),
                "design_checkpoint_sha256": actual_hash,
                "lock_file": str(args.lock),
                "config_name": config_name,
                "config": values,
                "search_summary": designer.last_search_summary,
                "internal_summary": candidate_summary(records),
                "candidates": [record.to_dict() for record in records],
            }
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output)
            print(
                f"SEARCH_COMPLETE seed={seed} target={target} "
                f"summary={json.dumps(designer.last_search_summary, sort_keys=True)}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    complete = args.output_root / "full229m_formal_search.complete"
    complete.touch()
    write_progress(progress_path, stage="formal_search_complete", complete=str(complete))
    print(f"FORMAL_SEARCH_COMPLETE marker={complete}", flush=True)


if __name__ == "__main__":
    main()
