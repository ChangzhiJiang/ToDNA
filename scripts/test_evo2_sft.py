#!/usr/bin/env python3
"""Fast checks for Evo 2 SFT data orientation, tokenization, and losses."""

from __future__ import annotations

import torch

from evo2_nabla_dna.sft import (
    BASE_TOKEN_IDS,
    causal_lm_losses,
    deterministic_orientation,
    reverse_complement,
    tokenize_sequences,
)


def main() -> None:
    assert reverse_complement("AACGT") == "ACGTT"
    sequence = "ACGT" * 50
    assert deterministic_orientation(sequence, 7) in {sequence, reverse_complement(sequence)}
    tokens = tokenize_sequences([sequence, reverse_complement(sequence)], "cpu")
    assert tokens.shape == (2, 200)
    logits = torch.full((2, 200, 512), -8.0, requires_grad=True)
    for token_id in BASE_TOKEN_IDS:
        logits.data[..., token_id] = 0.0
    for position in range(199):
        logits.data[:, position, tokens[:, position + 1]] = 8.0
    losses = causal_lm_losses(logits, tokens)
    assert float(losses["full_vocab_nll"]) < 0.01
    assert float(losses["dna_conditional_nll"]) < 0.01
    losses["full_vocab_nll"].backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    print("Evo 2 SFT checks passed")


if __name__ == "__main__":
    main()
