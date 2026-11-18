"""Shared helpers for formal TTO, factorial and autoregressive RL entry points.

Extracted unchanged from the former launchers. No experiment is launched when
this module is imported; seeds, rewards and search defaults remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from src import DtoConfig, SearchConfig, SequenceConstraints
from src.sft import BASE_TOKEN_IDS, tokenize_sequences



TASKS = ("k562", "hepg2", "sknsh")


PROJECT_ORDER = ("k562", "hepg2", "sknsh")


RANGES = {
    "k562": (-5.857445, 10.781755),
    "hepg2": (-6.051336, 10.992575),
    "sknsh": (-7.283977, 12.888308),
}


def load_pool(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = [str(value).upper() for value in payload["sequences"]]
    if len(sequences) != 768 or len(set(sequences)) != 768:
        raise ValueError("pool must contain 768 unique sequences")
    if any(len(value) != 200 or set(value) - set("ACGT") for value in sequences):
        raise ValueError("pool contains invalid 200-bp DNA")
    return sequences


def load_starts(path: Path, pool: set[str]) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = [str(item["sequence"]) for item in payload["sequences"]]
    if len(sequences) != 128 or len(set(sequences)) != 128:
        raise ValueError(f"starting set is not 128 unique sequences: {path}")
    if not set(sequences) <= pool:
        raise ValueError(f"starting set is not a subset of the shared pool: {path}")
    return sequences


class CtrlStyleDeltaR(nn.Module):
    def __init__(self, predictor: nn.Module, target: str):
        super().__init__()
        self.predictor = predictor
        self.target_index = PROJECT_ORDER.index(target)
        self.register_buffer(
            "low", torch.tensor([RANGES[name][0] for name in PROJECT_ORDER])
        )
        self.register_buffer(
            "high", torch.tensor([RANGES[name][1] for name in PROJECT_ORDER])
        )

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        normalized = (self.predictor(q) - self.low) / (self.high - self.low)
        off = [index for index in range(3) if index != self.target_index]
        return normalized[:, self.target_index] - normalized[:, off].mean(dim=1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def disable_flash_attention(model: torch.nn.Module) -> None:
    from vortex.model.attention import CrossAttention, SelfAttention

    replaced = 0
    for module in model.modules():
        mha = getattr(module, "inner_mha_cls", None)
        if mha is None or not hasattr(mha, "inner_attn"):
            continue
        self_attention = mha.inner_attn
        cross_attention = mha.inner_cross_attn
        mha.use_flash_attn = False
        mha.inner_attn = SelfAttention(
            causal=self_attention.causal,
            softmax_scale=self_attention.softmax_scale,
            attention_dropout=self_attention.drop.p,
        )
        mha.inner_cross_attn = CrossAttention(
            causal=cross_attention.causal,
            softmax_scale=cross_attention.softmax_scale,
            attention_dropout=cross_attention.drop.p,
        )
        replaced += 1
    if replaced == 0:
        raise RuntimeError("no Vortex attention blocks were found")


def token_base_indices(tokens: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(tokens)
    matched = torch.zeros_like(tokens, dtype=torch.bool)
    for index, token_id in enumerate(BASE_TOKEN_IDS):
        mask = tokens == token_id
        result[mask] = index
        matched |= mask
    if not bool(matched.all()):
        raise ValueError("tokens contain non-ACGT values")
    return result


def one_hot_sequences(sequences: list[str], device: torch.device | str) -> torch.Tensor:
    tokens = tokenize_sequences(sequences, device)
    return torch.nn.functional.one_hot(
        token_base_indices(tokens), num_classes=4
    ).float()


def build_search_config(
    values: dict[str, object], num_initial: int, top_k: int
) -> SearchConfig:
    prior_weight = float(values.get("prior_weight", 0.5))
    return SearchConfig(
        num_initial=num_initial,
        top_k=top_k,
        dto=DtoConfig(
            steps=int(values.get("steps", 5)),
            learning_rate=float(values.get("learning_rate", 0.1)),
            temperature=float(values.get("temperature", 0.7)),
            reward_weight=float(values.get("reward_weight", 1.0)),
            prior_weight=prior_weight,
            gradient_clip_norm=float(values.get("gradient_clip_norm", 5.0)),
            initial_sequence_margin=float(values.get("initial_margin", 0.25)),
            trust_region_weight=float(values.get("trust_region_weight", 1.0)),
            edit_penalty_weight=float(values.get("edit_penalty_weight", 2.0)),
            max_edits=(
                int(values["max_edits"])
                if values.get("max_edits") is not None
                else None
            ),
            initial_logit_noise_std=float(values.get("trajectory_noise_std", 0.0)),
            initial_logit_noise_seed=(
                int(values["trajectory_seed_base"])
                if values.get("trajectory_seed_base") is not None
                else None
            ),
            audit_hashes=bool(values.get("audit_hashes", False)),
        ),
        constraints=SequenceConstraints(
            min_gc=float(values.get("min_gc", 0.30)),
            max_gc=float(values.get("max_gc", 0.70)),
            max_homopolymer=int(values.get("max_homopolymer", 6)),
            max_evo2_nll=(
                float(values["max_evo2_nll"])
                if values.get("max_evo2_nll") is not None
                else None
            ),
        ),
        prior_penalty=float(values.get("ranking_prior_penalty", prior_weight)),
    )
