#!/usr/bin/env python3
"""Build a frozen 3-mer reference from the shared Design pools only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sequences(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = payload.get("sequences", payload) if isinstance(payload, dict) else payload
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"no sequences in {path}")
    return [str(value).upper() for value in sequences]


def profile(sequences: list[str]) -> list[int]:
    alphabet = "ACGT"
    index = {a + b + c: i for i, (a, b, c) in enumerate(
        (a, b, c) for a in alphabet for b in alphabet for c in alphabet
    )}
    counts = [0] * 64
    for sequence in sequences:
        if set(sequence) - set(alphabet) or len(sequence) < 3:
            raise ValueError("invalid DNA sequence")
        for start in range(len(sequence) - 2):
            counts[index[sequence[start : start + 3]]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sequences = [sequence for path in args.pool for sequence in load_sequences(path)]
    counts = profile(sequences)
    payload = {
        "protocol": "todna_shared_pool_3mer_reference_v1",
        "source_pools": [str(path) for path in args.pool],
        "source_pool_sha256": {str(path): sha256(path) for path in args.pool},
        "num_sequences": len(sequences),
        "targets": {target: counts for target in ("k562", "hepg2", "sknsh")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "num_sequences": len(sequences)}))


if __name__ == "__main__":
    main()
