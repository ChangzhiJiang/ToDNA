#!/usr/bin/env python3
"""Compute DNA-CRAFT-aligned 3-mer and diversity metrics without model calls."""

from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


BASES = "ACGT"
CELL_NAMES = ("K562", "HepG2", "SK-N-SH")
ACTIVITY_COLUMNS = ("K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC")
TARGET_FILES = {"K562": "k562", "HepG2": "hepg2", "SK-N-SH": "sknsh"}
KMERS = tuple("".join(bases) for bases in itertools.product(BASES, repeat=3))


def min_gap(activity: list[float], target_index: int) -> float:
    return activity[target_index] - max(
        value for index, value in enumerate(activity) if index != target_index
    )


def kmer_counts(sequences: list[str]) -> list[int]:
    counts: Counter[str] = Counter()
    for sequence in sequences:
        counts.update(sequence[index : index + 3] for index in range(len(sequence) - 2))
    return [counts[kmer] for kmer in KMERS]


def pearson(first: list[int], second: list[int]) -> float:
    first_mean = statistics.mean(first)
    second_mean = statistics.mean(second)
    covariance = sum(
        (a - first_mean) * (b - second_mean) for a, b in zip(first, second)
    )
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    if not first_scale or not second_scale:
        raise ValueError("3-mer count vector has zero variance")
    return covariance / (first_scale * second_scale)


def positional_entropy(sequences: list[str]) -> float:
    values = []
    for position in zip(*sequences):
        counts = Counter(position)
        entropy = 0.0
        for base in BASES:
            probability = counts[base] / len(position)
            if probability:
                entropy -= probability * math.log2(probability)
        values.append(entropy)
    return statistics.mean(values)


def build_references(
    table: Path, sequence_length: int, keep: int
) -> tuple[dict[str, list[tuple[float, str]]], int]:
    heaps: dict[str, list[tuple[float, str]]] = {name: [] for name in CELL_NAMES}
    valid_rows = 0
    with table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sequence = row["sequence"].strip().upper()
            if len(sequence) != sequence_length or set(sequence) - set(BASES):
                continue
            activity = [float(row[column]) for column in ACTIVITY_COLUMNS]
            valid_rows += 1
            for target_index, target in enumerate(CELL_NAMES):
                item = (min_gap(activity, target_index), sequence)
                heap = heaps[target]
                if len(heap) < keep:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
    return heaps, valid_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gosai-table", type=Path, required=True)
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--reference-fraction", type=float, default=0.001)
    parser.add_argument("--max-reference-candidates", type=int, default=5000)
    args = parser.parse_args()

    heaps, valid_rows = build_references(
        args.gosai_table, args.sequence_length, args.max_reference_candidates
    )
    reference_count = max(1, math.ceil(valid_rows * args.reference_fraction))
    if reference_count > args.max_reference_candidates:
        raise ValueError("increase --max-reference-candidates")

    result: dict[str, Any] = {
        "protocol": "DNA-CRAFT section 3.3 sequence-level metric alignment",
        "reference_definition": (
            "Top 99.9th percentile of real Gosai MPRA sequences by true MinGap "
            "for each target; overlapping aggregate 3-mer counts."
        ),
        "gosai_table": str(args.gosai_table),
        "valid_reference_rows": valid_rows,
        "reference_fraction": args.reference_fraction,
        "reference_count": reference_count,
        "targets": {},
    }
    for target in CELL_NAMES:
        reference = sorted(heaps[target], reverse=True)[:reference_count]
        reference_sequences = [sequence for _, sequence in reference]
        slug = TARGET_FILES[target]
        candidate_path = next(args.lock_root.glob(f"{slug}_seed*_top*.json"))
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        generated = [record["optimized_sequence"] for record in payload["candidates"]]
        result["targets"][slug] = {
            "target": target,
            "generated_count": len(generated),
            "reference_count": len(reference_sequences),
            "reference_mingap_threshold": min(score for score, _ in reference),
            "three_mer_pearson": pearson(
                kmer_counts(generated), kmer_counts(reference_sequences)
            ),
            "diversity_mean_positional_shannon_bits": positional_entropy(generated),
            "reference_diversity_mean_positional_shannon_bits": positional_entropy(
                reference_sequences
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
