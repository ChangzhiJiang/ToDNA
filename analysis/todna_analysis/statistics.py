"""Seed-level summaries with explicit identity and completeness checks."""
from collections import defaultdict
import math
from statistics import mean, stdev


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
