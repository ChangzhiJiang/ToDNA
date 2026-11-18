#!/usr/bin/env python3
"""Unit tests for custom Vortex LoRA injection without loading Evo 2."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn

from src.lora import (
    Evo2LoraConfig,
    adapter_state,
    inject_lora,
    load_lora_adapter,
    lora_parameters,
    restore_adapter_state,
    save_lora_adapter,
    trainable_parameter_summary,
)


class TupleLinear(nn.Linear):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return super().forward(inputs), None


class TinyVortex(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projections = TupleLinear(4, 12, bias=False)
        self.out_filter_dense = nn.Linear(4, 4)
        self.untouched = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected, _ = self.projections(inputs)
        return self.out_filter_dense(projected[..., :4]) + self.untouched(inputs)


def main() -> None:
    torch.manual_seed(7)
    model = TinyVortex()
    inputs = torch.randn(2, 3, 4)
    expected = model(inputs).detach()
    config = Evo2LoraConfig(rank=2, alpha=4, dropout=0)
    names = inject_lora(model, config)
    assert names == ["projections", "out_filter_dense"]
    assert torch.equal(expected, model(inputs))
    summary = trainable_parameter_summary(model)
    assert summary["trainable_parameters"] == sum(
        parameter.numel() for parameter in lora_parameters(model)
    )
    assert not any(parameter.requires_grad for parameter in model.untouched.parameters())

    loss = model(inputs).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in lora_parameters(model))

    state = {key: value.clone() for key, value in adapter_state(model).items()}
    for parameter in lora_parameters(model):
        parameter.data.zero_()
    restore_adapter_state(model, state)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "adapter.pt"
        save_lora_adapter(model, path, config, {"test": True})
        restored = TinyVortex()
        restored.load_state_dict(
            {
                key.replace(".base", ""): value
                for key, value in model.state_dict().items()
                if ".lora_" not in key
            },
            strict=True,
        )
        loaded_config, metadata = load_lora_adapter(restored, path)
        assert loaded_config == config
        assert metadata == {"test": True}
        assert torch.allclose(model(inputs), restored(inputs))

    filtered = TinyVortex()
    try:
        inject_lora(
            filtered,
            Evo2LoraConfig(rank=2, alpha=4, dropout=0, block_indices=(31,)),
        )
    except RuntimeError as error:
        assert "no Evo 2 projection modules" in str(error)
    else:
        raise AssertionError("block filtering should reject the non-block toy model")
    print("Evo 2 LoRA checks passed")


if __name__ == "__main__":
    main()
