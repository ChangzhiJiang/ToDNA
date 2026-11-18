#!/usr/bin/env python3
"""Recompute tables from supplied numerical records, without GPUs or downloads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import math
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent

SEEDS = (20260920, 20260921, 20260922)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Missing or nonfinite measurement")
    return value


def summary(values):
    values = [finite(v) for v in values]
    if not values:
        raise ValueError("Empty measurement group")
    return {"n": len(values), "mean": mean(values),
            "sample_sd": stdev(values) if len(values) > 1 else None}


def aggregate(rows, keys, metrics, expected_seeds):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in keys)
        seed = int(row["seed"])
        if seed in groups[key]:
            raise ValueError(f"Duplicate seed {seed} in {key}")
        groups[key][seed] = row
    if not groups:
        raise ValueError("No groups")
    result = []
    for key, group in sorted(groups.items()):
        if set(group) != set(expected_seeds):
            raise ValueError(f"Incomplete or unexpected seed set for {key}")
        for metric in metrics:
            result.append({**dict(zip(keys, key)), "metric": metric,
                           **summary(group[s][metric] for s in sorted(group))})
    return result


def pairs(rows, arm_key, left, right, identity_keys=(), expected_seeds=()):
    """Return validated pairs and exclusions; never silently drop a missing arm."""
    grouped = defaultdict(dict)
    for row in rows:
        seed, arm = int(row["seed"]), row[arm_key]
        if arm not in (left, right):
            raise ValueError(f"Unexpected arm: {arm}")
        if arm in grouped[seed]:
            raise ValueError(f"Duplicate arm for seed {seed}")
        grouped[seed][arm] = row
    if set(grouped) != set(expected_seeds):
        raise ValueError("Incomplete or unexpected paired seed set")
    accepted, excluded = [], []
    for seed, arms in sorted(grouped.items()):
        if set(arms) != {left, right}:
            raise ValueError(f"Missing paired arm for seed {seed}")
        a, b = arms[left], arms[right]
        mismatches = [k for k in identity_keys
                      if not a.get(k) or not b.get(k) or a[k] != b[k]]
        for arm, row in arms.items():
            if "cache_adapter_id" in row:
                if row["cache_adapter_id"] != row.get("run_adapter_id"):
                    mismatches.append(f"{arm}:cache_adapter_id!=run_adapter_id")
        if mismatches:
            excluded.append({"seed": seed, "reason": "; ".join(mismatches)})
        else:
            accepted.append((seed, a, b))
    return accepted, excluded


def cost_totals(rows, expected_seeds):
    grouped = defaultdict(dict)
    for row in rows:
        method, seed = row["method"], int(row["seed"])
        if seed in grouped[method]:
            raise ValueError("Duplicate method/seed cost record")
        grouped[method][seed] = row
    if set(grouped) != {"ToDNA", "PPO", "GRPO"}:
        raise ValueError("Unexpected cost methods")
    result = []
    for method in ("ToDNA", "PPO", "GRPO"):
        if set(grouped[method]) != set(expected_seeds):
            raise ValueError(f"Missing cost seed for {method}")
        group = list(grouped[method].values())
        for r in group:
            for k in ("allocated_gpu_hours", "run_elapsed_seconds", "peak_allocated_bytes"):
                if finite(r[k]) < 0:
                    raise ValueError("Negative resource measurement")
        result.append({"method": method, "n_seeds": len(group),
                       "total_allocated_gpu_hours": sum(finite(r["allocated_gpu_hours"]) for r in group),
                       "cumulative_runtime_hours": sum(finite(r["run_elapsed_seconds"]) for r in group) / 3600,
                       "peak_allocated_GiB_per_GPU": max(finite(r["peak_allocated_bytes"]) for r in group) / 2**30})
    return result


def load(name):
    path = ROOT / "data" / name
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_inputs():
    manifest = json.loads((ROOT / "provenance/INPUT_SHA256.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input changed: {name}")
    return len(manifest)


def display(value):
    if value is None:
        return "NA"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def write_table(out, name, rows, caption):
    if not rows:
        raise ValueError(f"Empty table: {name}")
    columns = list(rows[0])
    with (out / (name + ".csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    text = caption + "\n\n| " + " | ".join(columns) + " |\n"
    text += "| " + " | ".join("---" for _ in columns) + " |\n"
    text += "\n".join("| " + " | ".join(display(row[c]) for c in columns) + " |" for row in rows) + "\n"
    (out / (name + ".md")).write_text(text)


def run(out):
    verified = verify_inputs()
    out.mkdir(parents=True, exist_ok=True)
    tables = []

    def emit(name, rows, caption):
        write_table(out, name, rows, caption)
        tables.append({"table": name, "rows": len(rows), "interpretation": caption})

    costs = cost_totals(load("cost_per_seed.csv"), SEEDS)
    emit("costs", costs, "HepG2, three seeds. Allocated GPU-hours and cumulative stage runtime; maximum single-device allocated memory. Not a same-hardware speed comparison.")
    phase_totals = cost_totals(load("cost_phase_records.json"), SEEDS)
    if costs != phase_totals:
        raise ValueError("Phase-record process totals differ from cost CSV")

    emit("six_task_summary", aggregate(load("six_task_per_seed.csv"), ["family", "task", "method"],
         ["evaluation_mean_margin", "evaluation_min_gap", "positional_entropy_bits", "pairwise_hamming_fraction"], SEEDS),
         "Original six-task archive: mean and sample SD across three seeds, separately for each task and method. Mean margin and MinGap here are normalized scores.")

    nearest = load("nearest_baselines_per_run.csv")
    if len(nearest) != 108 or any(int(r["n"]) != 256 or int(r["design_sequence_calls"]) != 5280 for r in nearest):
        raise ValueError("Unexpected nearest-baseline budget or output count")
    emit("nearest_baselines", aggregate(nearest, ["family", "task", "arm"],
         ["evaluation_mean_margin", "evaluation_min_gap", "true_edit_count", "retained_pairwise_hamming_fraction"],
         range(2026110621, 2026110627)),
         "ToDNA/FastSeqProp/Ledidi: six search seeds on a fixed pool, 5,280 Design calls and 256 retained outputs per run. These are not six independent adapter-training or pool-generation replicates.")

    factorial = load("factorial_cells.json")
    if len(factorial) != 24 or any(int(r["n"]) != 256 for r in factorial):
        raise ValueError("Unexpected factorial cell count")
    emit("factorial_descriptive", aggregate(factorial,
         ["pool_generation", "logit_initialization", "kl_reference"],
         ["evaluation_min_gap_mean", "fullmpra_min_gap_mean", "three_mer_pearson", "edits_mean"], SEEDS),
         "DESCRIPTIVE ONLY: archived 24 cells. Predictor/adapter identity is not verified consistently across the complete factorial; do not interpret this table as strictly controlled main or interaction effects. MinGap here is raw, not normalized.")

    keys = ("design_checkpoint_sha256", "adapter_sha256", "pool_sha256", "starts_sha256", "logit_cache_sha256")
    reference = load("reference_pairs.json")
    accepted, exclusions = pairs(reference, "kl_reference", "base", "lora", keys, SEEDS)
    if exclusions:
        raise ValueError("Reference-only identity mismatch")
    reference_results = []
    for metric in ("evaluation_min_gap_mean", "fullmpra_min_gap_mean", "three_mer_pearson", "edits_mean"):
        for label, values in [
            ("Base", [a[metric] for _, a, b in accepted]),
            ("LoRA", [b[metric] for _, a, b in accepted]),
            ("LoRA_minus_Base", [finite(b[metric])-finite(a[metric]) for _, a, b in accepted])]:
            reference_results.append({"contrast": label, "metric": metric, **summary(values)})
    emit("reference_only", reference_results, "Identity-verified reference-only pairs: Base-generated pool and Base initialization in both arms; both retain an Evo2 KL reference. SD of differences is computed from paired differences.")

    accepted, exclusions = pairs(load("start_pairs.json"), "arm", "random", "ranked",
        ("pool_id", "design_id", "run_adapter_id", "cache_adapter_id", "queries", "retained"), SEEDS)
    rows = [{"seed": s, "random": finite(a["final_reward"]), "ranked": finite(b["final_reward"]),
             "ranked_minus_random": finite(b["final_reward"])-finite(a["final_reward"])} for s,a,b in accepted]
    emit("start_selection_pairs", rows, "Design reward for the verified paired subset. Seed 20260920 is excluded because the ranked arm uses an inconsistent cache adapter; see pairing_exclusions.json. Two pairs cannot establish general equivalence.")
    (out / "pairing_exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")

    emit("regularizers", aggregate(load("regularizers.csv"), ["target", "arm"],
         ["mean_reward_gain", "mean_hamming_edits"], (20260824, 20260825, 20260826)),
         "KL/Edit/None at the archived coefficients: Design reward gain and edits, three seeds. Not an independent-evaluator result or an edit-matched coefficient sweep.")

    retained = load("retained_pairs.csv")
    if len(retained) != 3 or {int(r["seed"]) for r in retained} != set(SEEDS):
        raise ValueError("Incomplete retained-pair table")
    rows = []
    for metric in ("reward", "gc", "positional_entropy", "pairwise_hamming", "three_mer_entropy"):
        for label in ("pre", "post", "post_minus_pre"):
            values = [finite(r[metric+"_"+label]) if label != "post_minus_pre"
                      else finite(r[metric+"_post"])-finite(r[metric+"_pre"]) for r in retained]
            rows.append({"metric": metric, "stage": label, **summary(values)})
    emit("retained_before_after", rows, "Three seed summaries conditional on the ultimately retained trajectories. Before/after cohort is not the full proposal pool. Input values are rounded archived summaries.")

    d3 = load("d3_pairs.csv")
    for row in d3:
        delta = finite(row["D3_plus_ToDNA"])-finite(row["proposal_only"])
        if abs(delta-finite(row["paired_difference"])) > 1e-12:
            raise ValueError("D3 paired difference inconsistent")
    emit("d3_pairs", aggregate(d3, ["target"], ["proposal_only", "D3_plus_ToDNA", "paired_difference"], SEEDS),
         "Three paired seeds. Final normalized evaluation reward, not raw MinGap. Adding TTO uses additional compute and retains Evo2 initialization/reference; this is not a complete Evo2 replacement.")

    emit("gpa_svdd_cohorts", aggregate(load("gpa_svdd_cohorts.csv"), ["method", "target"],
         ["evaluation_mingap_mean_all_768", "top256_mingap_mean"], SEEDS),
         "DESCRIPTIVE ONLY: full 768 and Evaluation-selected Top-256 are distinct cohorts. Methods have different Design-call budgets; Evaluation-selected results are not blind-evaluation rankings against Design-selected methods.")
    (out / "REPORT_INDEX.json").write_text(json.dumps({"verified_inputs": verified, "tables": tables}, indent=2)+"\n")
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    tables = run(args.output)
    print(f"Verified inputs and wrote {len(tables)} tables to {args.output}")
