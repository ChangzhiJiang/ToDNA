#!/usr/bin/env python3
"""Precompute Base and LoRA teacher-forced logits for one immutable start set."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from evo2_nabla_dna import Evo2TestTimeDesigner
from evo2_nabla_dna.profiling import CostProfiler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sequences(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_values = payload.get("sequences", payload)
    if not isinstance(raw_values, list):
        raise ValueError(f"invalid sequence list: {path}")
    values = [
        value["sequence"] if isinstance(value, dict) and "sequence" in value else value
        for value in raw_values
    ]
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"invalid sequence list: {path}")
    if len(values) != 128 or len(set(values)) != 128:
        raise ValueError("factorial contexts require 128 unique starts")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    sequences = load_sequences(args.starts)
    if any(len(sequence) != args.length or set(sequence) - set("ACGT") for sequence in sequences):
        raise ValueError("starts contain invalid sequence length or alphabet")
    profiler = CostProfiler(method="ToDNA", protocol="strict_hepg2_lora_pool_init_ref_factorial_v1")
    contexts: dict[str, torch.Tensor] = {}
    for name, adapter in (("base", None), ("lora", args.adapter)):
        with profiler.phase(f"model_load_{name}"):
            designer = Evo2TestTimeDesigner(model_name="evo2_7b", lora_adapter=adapter)
            designer.dto.disable_flash_attention()
        with profiler.phase(f"teacher_forced_logits_{name}", teacher_forced_sequences=len(sequences)):
            rows = [
                designer.dto.logits_for_sequence(sequence, initial_sequence_margin=0.25)
                .detach()
                .cpu()
                .float()
                for sequence in sequences
            ]
            contexts[name] = torch.cat(rows, dim=0)
        del designer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "protocol": "strict_hepg2_lora_pool_init_ref_factorial_v1",
        "starts": str(args.starts),
        "starts_sha256": sha256(args.starts),
        "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter),
        "sequence_length": args.length,
        "n_starts": len(sequences),
        "sequences": sequences,
        "logits": contexts,
        "cost_audit": profiler.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output),
        "n_starts": len(sequences),
        "cost_audit": payload["cost_audit"],
    }, indent=2))


if __name__ == "__main__":
    main()
