#!/usr/bin/env python3
"""Portable entry for the archived family-specific LoRA launcher.

The original launcher substituted a 200/250-bp reader before calling the
unchanged trainer. This wrapper preserves that behavior without a server queue.
It validates existing split hashes; it does not recreate or alter the splits.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys


def build_plan(family, data_root, output_dir):
    data_root, output_dir = Path(data_root), Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output already exists; refusing to overwrite an adapter run")
    manifest = json.loads((data_root / "complete.json").read_text())
    if manifest["family"] != family:
        raise ValueError("Data manifest family mismatch")
    for split in ("train", "validation"):
        path = data_root / "lora" / (split + ".jsonl")
        expected = manifest["partitions"]["lora_design_" + split]["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Split hash mismatch: {split}")
    seed = 20260914 if family == "enhancer" else 20260915
    return {
        "family": family, "sequence_length": 200 if family == "enhancer" else 250,
        "arguments": ["--data-root", str(data_root / "lora"), "--output-dir", str(output_dir),
                      "--seed", str(seed), "--max-steps", "2000", "--rank", "8",
                      "--alpha", "16", "--last-n-blocks", "8", "--learning-rate", "0.0001",
                      "--batch-size", "1", "--gradient-accumulation", "8",
                      "--validate-every", "250", "--validation-sequences", "256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("enhancer", "promoter"), required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Family directory containing complete.json and lora/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--describe-only", action="store_true", help="Validate hashes and print the plan without loading a model")
    args = parser.parse_args()
    plan = build_plan(args.family, args.data_root, args.output_dir)
    if args.describe_only:
        print(json.dumps(plan, indent=2))
    else:
        import train_evo2_lora_sft as core
        core.read_sequence_jsonl = functools.partial(core.read_sequence_jsonl, length=plan["sequence_length"])
        sys.argv = [sys.argv[0], *plan["arguments"]]
        core.main()
