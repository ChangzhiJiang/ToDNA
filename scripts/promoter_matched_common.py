"""Shared immutable-oracle utilities for the public promoter benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from src.predictor import GreluEnformerActivity


TASKS = ("JURKAT", "K562", "THP1")
RANGES = {
    "JURKAT": (-5.574782, 8.555965),
    "K562": (-4.088671, 10.781755),
    "THP1": (-7.271035, 6.797082),
}


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


def load_pool(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = [str(value).upper() for value in payload["sequences"]]
    if len(sequences) != 768 or len(set(sequences)) != 768:
        raise ValueError("pool must contain 768 unique sequences")
    if any(len(value) != 250 or set(value) - set("ACGT") for value in sequences):
        raise ValueError("pool contains invalid 250-bp DNA")
    return sequences


def load_starts(path: Path, pool: set[str]) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = [str(item["sequence"]) for item in payload["sequences"]]
    if len(sequences) != 128 or len(set(sequences)) != 128:
        raise ValueError(f"starting set is not 128 unique sequences: {path}")
    if not set(sequences) <= pool:
        raise ValueError(f"starting set is not a subset of the shared pool: {path}")
    return sequences


def load_promoter_checkpoint(
    checkpoint: Path, enformer_state_dict: Path, device: torch.device | str
) -> tuple[GreluEnformerActivity, tuple[str, ...]]:
    import grelu.resources
    from grelu.lightning import LightningModel

    state_dict = os.path.realpath(enformer_state_dict)

    class LocalEnformerArtifact:
        def download(self, root):
            destination = os.path.join(root, "human.h5")
            if not os.path.exists(destination):
                os.symlink(state_dict, destination)
            return root

    original_get_artifact = grelu.resources.get_artifact
    grelu.resources.get_artifact = lambda *args, **kwargs: LocalEnformerArtifact()
    try:
        model = LightningModel.load_from_checkpoint(checkpoint, map_location=device)
    finally:
        grelu.resources.get_artifact = original_get_artifact
    task_names = tuple(str(value) for value in model.data_params["tasks"]["name"])
    if task_names != TASKS:
        raise ValueError(f"unexpected task order: {task_names}")
    return GreluEnformerActivity(model).to(device), task_names


def padded_scores(predictor, q: torch.Tensor, input_length: int = 2114) -> torch.Tensor:
    if q.shape[1] > input_length:
        raise ValueError("oracle input is shorter than sequence")
    left = (input_length - q.shape[1]) // 2
    padded = torch.zeros(q.shape[0], input_length, 4, device=q.device, dtype=q.dtype)
    padded[:, left:left + q.shape[1]] = q
    return predictor(padded)
