from __future__ import annotations

import torch
from torch import nn

from evo2_nabla_dna.rewards import (
    MultiCellActivityReward,
    NormalizedDeltaRReward,
    NormalizedTargetActivityReward,
)


class TinyActivityPredictor(nn.Module):
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        base_frequencies = q.mean(dim=1)
        target = 2.0 * base_frequencies[:, 0] - base_frequencies[:, 1]
        off_target_1 = base_frequencies[:, 1] + base_frequencies[:, 2]
        off_target_2 = base_frequencies[:, 3]
        return torch.stack([target, off_target_1, off_target_2], dim=-1)


def main() -> None:
    logits = torch.zeros(2, 16, 4, requires_grad=True)
    q = logits.softmax(dim=-1)
    reward = MultiCellActivityReward(
        TinyActivityPredictor(), target_index=0, off_target_indices=[1, 2]
    )(q)
    reward.backward()

    assert torch.isfinite(reward)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    print(f"reward={float(reward):.6f} gradient_norm={float(logits.grad.norm()):.6f}")

    delta_logits = torch.zeros(2, 16, 4, requires_grad=True)
    delta_q = delta_logits.softmax(dim=-1)
    delta_reward = NormalizedDeltaRReward(
        TinyActivityPredictor(),
        target_index=0,
        off_target_indices=[1, 2],
        activity_min=[0.0, 0.0, 0.0],
        activity_max=[2.0, 2.0, 2.0],
    )(delta_q)
    delta_reward.backward()

    with torch.no_grad():
        raw = TinyActivityPredictor()(delta_q)
        normalized = raw / 2.0
        expected = (
            normalized[:, 0] - normalized[:, [1, 2]].mean(dim=-1)
        ).mean()
    assert torch.allclose(delta_reward, expected)
    assert delta_logits.grad is not None
    assert torch.isfinite(delta_logits.grad).all()
    assert delta_logits.grad.abs().sum() > 0
    print(
        f"delta_r_reward={float(delta_reward):.6f} "
        f"gradient_norm={float(delta_logits.grad.norm()):.6f}"
    )

    target_logits = torch.zeros(2, 16, 4, requires_grad=True)
    target_q = target_logits.softmax(dim=-1)
    target_reward = NormalizedTargetActivityReward(
        TinyActivityPredictor(),
        target_index=0,
        activity_min=[0.0, 0.0, 0.0],
        activity_max=[2.0, 2.0, 2.0],
    )(target_q)
    target_reward.backward()
    with torch.no_grad():
        expected_target = (TinyActivityPredictor()(target_q)[:, 0] / 2.0).mean()
    assert torch.allclose(target_reward, expected_target)
    assert target_logits.grad is not None
    assert torch.isfinite(target_logits.grad).all()
    assert target_logits.grad.abs().sum() > 0
    print(
        f"absolute_target_reward={float(target_reward):.6f} "
        f"gradient_norm={float(target_logits.grad.norm()):.6f}"
    )


if __name__ == "__main__":
    main()
