#!/usr/bin/env python3
"""Run uncapped Evo2-PPO or Evo2-GRPO on a shared Design Top-128 enhancer set.

This is a contextual-bandit sequence-edit policy, not reward-weighted SFT. The
only trainable parameters are Evo 2 LoRA matrices. Evaluation, FullMPRA, FIMO,
and all external metrics are excluded until the Top-256 lock is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from evo2 import Evo2

from evo2_nabla_dna import load_activity_checkpoint
from evo2_nabla_dna.constraints import SequenceConstraints
from evo2_nabla_dna.lora import (
    adapter_state,
    load_lora_adapter,
    lora_parameters,
    materialize_autograd_tensors,
    restore_adapter_state,
    save_lora_adapter,
    set_lora_training,
)
from evo2_nabla_dna.rl import (
    BASE_TOKEN_IDS,
    action_log_probs,
    categorical_kl,
    clipped_policy_loss,
    edit_policy_logits,
    expected_edit_fraction,
    grpo_advantages,
    policy_entropy,
    ppo_advantages,
    sample_edit_actions,
)
from evo2_nabla_dna.sft import tokenize_sequences


TASKS = ("k562", "hepg2", "sknsh")
DISPLAY = {"k562": "K562", "hepg2": "HepG2", "sknsh": "SK-N-SH"}
RANGES = {
    "k562": (-5.857445, 10.781755),
    "hepg2": (-6.051336, 10.992575),
    "sknsh": (-7.283977, 12.888308),
}
BASES = "ACGT"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
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


def load_pool(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = [str(value).upper() for value in payload["sequences"]]
    if len(sequences) != 768 or len(set(sequences)) != 768:
        raise ValueError("pool must contain 768 unique sequences")
    if any(len(value) != 200 or set(value) - set(BASES) for value in sequences):
        raise ValueError("pool contains invalid 200-bp DNA")
    return set(sequences)


def load_starts(path: Path, pool: set[str]) -> tuple[list[str], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["sequences"]
    sequences = [str(item["sequence"]).upper() for item in records]
    if len(sequences) != 128 or len(set(sequences)) != 128:
        raise ValueError("shared start must contain 128 unique sequences")
    if not set(sequences) <= pool:
        raise ValueError("shared starts are not a subset of the 768 pool")
    # Ctrl-DNA stores 2*DeltaR_mean+1 as `target`; invert the affine transform.
    baselines = [(float(item["target"]) - 1.0) / 2.0 for item in records]
    return sequences, baselines


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


def decode_actions(actions: torch.Tensor) -> list[str]:
    return ["".join(BASES[index] for index in row) for row in actions.cpu().tolist()]


def one_hot_sequences(sequences: list[str], device: torch.device | str) -> torch.Tensor:
    tokens = tokenize_sequences(sequences, device)
    return torch.nn.functional.one_hot(
        token_base_indices(tokens), num_classes=4
    ).float()


class DesignDeltaR(torch.nn.Module):
    def __init__(self, predictor: torch.nn.Module, target: str):
        super().__init__()
        self.predictor = predictor
        self.target_index = TASKS.index(target)
        self.register_buffer("low", torch.tensor([RANGES[name][0] for name in TASKS]))
        self.register_buffer("high", torch.tensor([RANGES[name][1] for name in TASKS]))

    def per_sequence(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activity = self.predictor(q)
        normalized = (activity - self.low) / (self.high - self.low)
        off = [index for index in range(3) if index != self.target_index]
        reward = normalized[:, self.target_index] - normalized[:, off].mean(dim=1)
        return reward, activity


def score_sequences(
    reward: DesignDeltaR,
    sequences: list[str],
    device: torch.device | str,
    batch_size: int,
) -> tuple[list[float], list[list[float]]]:
    rewards: list[float] = []
    activities: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            q = one_hot_sequences(sequences[start : start + batch_size], device)
            values, raw = reward.per_sequence(q)
            rewards.extend(values.float().cpu().tolist())
            activities.extend(raw.float().cpu().tolist())
    return rewards, activities


def constraint_metrics(sequence: str) -> dict[str, float]:
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
    run = longest = 1
    for left, right in zip(sequence, sequence[1:]):
        run = run + 1 if left == right else 1
        longest = max(longest, run)
    return {"gc_fraction": float(gc), "max_homopolymer": float(longest)}


def round_state_indices(seed: int, round_index: int, count: int) -> list[int]:
    """Use every shared start exactly once in each consecutive pair of rounds."""
    if 128 % count:
        raise ValueError("states per round must divide 128")
    block = round_index // (128 // count)
    offset = round_index % (128 // count)
    order = list(range(128))
    random.Random(seed * 100_003 + block).shuffle(order)
    return order[offset * count : (offset + 1) * count]


def completed_lock(path: Path, top_k: int) -> bool:
    if not path.is_file():
        return False
    return len(json.loads(path.read_text(encoding="utf-8")).get("candidates", [])) == top_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("ppo", "grpo"), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--shared-start", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", choices=TASKS, default="hepg2")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--states-per-round", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--edit-weight", type=float, default=0.10)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--stay-bias", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ppo-baseline-momentum", type=float, default=0.8)
    parser.add_argument("--oracle-batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--model-name", default="evo2_7b")
    args = parser.parse_args()

    calls = args.rounds * args.states_per_round * args.group_size
    if calls != 15_360:
        raise ValueError(f"strict protocol requires 15360 oracle calls, got {calls}")
    if args.gradient_accumulation <= 0 or args.states_per_round % args.gradient_accumulation:
        raise ValueError("gradient accumulation must divide states per round")
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_dir = args.output_root / "formal_lock"
    lock_path = lock_dir / f"{args.target}_seed{args.seed}_top{args.top_k}.json"
    if completed_lock(lock_path, args.top_k):
        print("LORA_POLICY_ALREADY_COMPLETE", flush=True)
        return

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({
        "protocol": "evo2_rl_uncapped_table2_v1",
        "oracle_sequence_evaluations": calls,
        "selection_boundary": "Design oracle and hard constraints only",
        "posthoc_excluded_before_lock": ["Evaluation", "FullMPRA", "Agarwal", "accessibility", "FIMO"],
        "policy_definition": "Evo2 causal ACGT logits conditioned on each shared start, shifted to editable positions with a fixed stay bias",
        "ppo_advantage": "prompt-conditioned exponential-moving-average reward baseline, then batch standardization",
        "grpo_advantage": "within-start group reward standardization; no value model",
    })
    config["config_sha256"] = json_hash(config)
    write_json(args.output_root / "run_config.json", config)
    pool = load_pool(args.pool)
    starts, initial_baselines = load_starts(args.shared_start, pool)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    write_json(args.output_root / "progress.json", {"stage": "loading_evo2", "config": config})
    evo = Evo2(args.model_name)
    model = evo.model
    materialize_autograd_tensors(model)
    lora_config, lora_metadata = load_lora_adapter(model, args.adapter)
    initial_adapter = {key: value.clone() for key, value in adapter_state(model).items()}
    parameters = lora_parameters(model)
    set_lora_training(model, False)
    disable_flash_attention(model)
    device = model.block_idx_to_device[0]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    predictor = load_activity_checkpoint(str(args.design_checkpoint), device=device)
    reward_fn = DesignDeltaR(predictor, args.target).to(device)

    # The immutable SFT-LoRA policy is cached for all 128 shared states.
    reference_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for index, sequence in enumerate(starts):
            tokens = tokenize_sequences([sequence], device)
            outputs, _ = model(tokens)
            reference_logits.append(
                edit_policy_logits(outputs, tokens, args.stay_bias, args.temperature)
                .squeeze(0).half().cpu()
            )
            if (index + 1) % 16 == 0:
                print(f"REFERENCE_POLICY {index + 1}/128", flush=True)

    state_path = args.output_root / "train_state.pt"
    records_path = args.output_root / "candidates.jsonl"
    baselines = torch.tensor(initial_baselines, dtype=torch.float32)
    start_round = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["config_sha256"] != config["config_sha256"]:
            raise ValueError("resume configuration hash mismatch")
        restore_adapter_state(model, state["adapter_state"])
        optimizer.load_state_dict(state["optimizer"])
        baselines = state["baselines"].float()
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        start_round = int(state["next_round"])

    started = time.time()
    for round_index in range(start_round, args.rounds):
        selected_indices = round_state_indices(args.seed, round_index, args.states_per_round)
        rollout_rows = []
        flat_sequences: list[str] = []
        with torch.no_grad():
            for state_index in selected_indices:
                tokens = tokenize_sequences([starts[state_index]], device)
                base_indices = token_base_indices(tokens)
                outputs, _ = model(tokens)
                old_policy = edit_policy_logits(
                    outputs, tokens, args.stay_bias, args.temperature
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    args.seed * 1_000_003 + round_index * 1009 + state_index
                )
                actions = sample_edit_actions(
                    old_policy, base_indices, args.group_size, generator
                )
                sequences = decode_actions(actions.squeeze(0))
                old_log = action_log_probs(old_policy, actions).squeeze(0).cpu()
                reference = reference_logits[state_index].float().unsqueeze(0).to(device)
                ref_log = action_log_probs(reference, actions).squeeze(0).cpu()
                rollout_rows.append({
                    "state_index": state_index,
                    "actions": actions.squeeze(0).cpu(),
                    "old_log_probs": old_log,
                    "reference_action_nll": -ref_log.mean(dim=-1),
                    "sequences": sequences,
                })
                flat_sequences.extend(sequences)

        rewards, activities = score_sequences(
            reward_fn, flat_sequences, device, args.oracle_batch_size
        )
        reward_matrix = torch.tensor(rewards).reshape(
            args.states_per_round, args.group_size
        )
        if args.algorithm == "grpo":
            advantages = grpo_advantages(reward_matrix)
        else:
            advantages = ppo_advantages(
                reward_matrix, baselines[selected_indices]
            )
            momentum = args.ppo_baseline_momentum
            baselines[selected_indices] = (
                momentum * baselines[selected_indices]
                + (1.0 - momentum) * reward_matrix.mean(dim=1)
            )

        offset = 0
        with records_path.open("a", encoding="utf-8") as handle:
            for row_index, row in enumerate(rollout_rows):
                state_index = row["state_index"]
                for group_index, sequence in enumerate(row["sequences"]):
                    metrics = constraint_metrics(sequence)
                    edits = sum(a != b for a, b in zip(starts[state_index], sequence))
                    record = {
                        "round": round_index,
                        "state_index": state_index,
                        "group_index": group_index,
                        "initial_sequence": starts[state_index],
                        "optimized_sequence": sequence,
                        "design_delta_r": rewards[offset],
                        "design_activity": activities[offset],
                        "reference_action_nll": float(row["reference_action_nll"][group_index]),
                        "hamming_edits": edits,
                        "constraint_metrics": metrics,
                        "accepted": True,
                    }
                    handle.write(json.dumps(record) + "\n")
                    offset += 1

        diagnostics = []
        for epoch in range(args.update_epochs):
            optimizer.zero_grad(set_to_none=True)
            for row_index, row in enumerate(rollout_rows):
                state_index = row["state_index"]
                tokens = tokenize_sequences([starts[state_index]], device)
                state_bases = token_base_indices(tokens)
                outputs, _ = model(tokens)
                current_policy = edit_policy_logits(
                    outputs, tokens, args.stay_bias, args.temperature
                )
                actions = row["actions"].unsqueeze(0).to(device)
                current_log = action_log_probs(current_policy, actions)
                old_log = row["old_log_probs"].unsqueeze(0).to(device)
                advantage = advantages[row_index].unsqueeze(0).to(device)
                policy_loss, policy_diag = clipped_policy_loss(
                    current_log, old_log, advantage, args.clip_epsilon
                )
                reference = reference_logits[state_index].float().unsqueeze(0).to(device)
                kl = categorical_kl(current_policy, reference)
                edit = expected_edit_fraction(current_policy, state_bases)
                entropy = policy_entropy(current_policy)
                loss = (
                    policy_loss
                    + args.kl_weight * kl
                    + args.edit_weight * edit
                    - args.entropy_weight * entropy
                ) / args.gradient_accumulation
                loss.backward()
                diagnostics.append({
                    "policy_loss": float(policy_loss.detach()),
                    "reference_kl": float(kl.detach()),
                    "expected_edit_fraction": float(edit.detach()),
                    "entropy": float(entropy.detach()),
                    **{name: float(value) for name, value in policy_diag.items()},
                })
                if (row_index + 1) % args.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        mean_diag = {
            key: statistics.fmean(item[key] for item in diagnostics)
            for key in diagnostics[0]
        }
        event = {
            "round": round_index + 1,
            "rounds": args.rounds,
            "oracle_calls": (round_index + 1) * args.states_per_round * args.group_size,
            "reward_mean": float(reward_matrix.mean()),
            "reward_std": float(reward_matrix.std(unbiased=True)),
            "diagnostics": mean_diag,
            "elapsed_seconds": time.time() - started,
        }
        with (args.output_root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        atomic_torch_save(state_path, {
            "config_sha256": config["config_sha256"],
            "next_round": round_index + 1,
            "adapter_state": adapter_state(model),
            "optimizer": optimizer.state_dict(),
            "baselines": baselines,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        })
        write_json(args.output_root / "progress.json", {"stage": "training", **event})
        print(f"POLICY_ROUND {json.dumps(event)}", flush=True)

    save_lora_adapter(
        model,
        args.output_root / "final_adapter.pt",
        lora_config,
        {
            "protocol": config["protocol"],
            "algorithm": args.algorithm,
            "seed": args.seed,
            "target": args.target,
            "source_adapter": str(args.adapter),
            "source_adapter_sha256": sha256(args.adapter),
            "source_adapter_metadata": lora_metadata,
        },
    )
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    by_sequence: dict[str, dict] = {}
    for record in records:
        if not record["accepted"]:
            continue
        sequence = record["optimized_sequence"]
        previous = by_sequence.get(sequence)
        if previous is None or record["design_delta_r"] > previous["design_delta_r"]:
            by_sequence[sequence] = record
    ordered = sorted(by_sequence.values(), key=lambda item: (-item["design_delta_r"], item["optimized_sequence"]))
    if len(ordered) < args.top_k:
        raise RuntimeError(f"only {len(ordered)} unique valid candidates; need {args.top_k}")
    selected = ordered[: args.top_k]
    candidates = []
    for rank, item in enumerate(selected, start=1):
        candidates.append({
            "rank": rank,
            "initial_sequence": item["initial_sequence"],
            "optimized_sequence": item["optimized_sequence"],
            "optimized_reward": item["design_delta_r"],
            "delta_r_mean": item["design_delta_r"],
            "design_activity": item["design_activity"],
            "reference_action_nll": item["reference_action_nll"],
            "hamming_edits": item["hamming_edits"],
            "nearest_initial_hamming": item["hamming_edits"],
            "generation_round": item["round"],
            "constraint_metrics": item["constraint_metrics"],
        })
    search_summary = {
        "attempted": calls,
        "unique_generated": len({item["optimized_sequence"] for item in records}),
        "hard_constraint_valid_unique": len(ordered),
        "accepted": len(ordered),
        "locked": len(candidates),
        "mean_delta_r_mean": statistics.fmean(item["delta_r_mean"] for item in candidates),
        "mean_edits": statistics.fmean(item["hamming_edits"] for item in candidates),
    }
    lock = {
        **config,
        "protocol": "evo2_rl_uncapped_table2_lock_v1",
        "seed": args.seed,
        "target": DISPLAY[args.target],
        "target_slug": args.target,
        "pool": str(args.pool),
        "pool_sha256": sha256(args.pool),
        "shared_top128": str(args.shared_start),
        "shared_top128_sha256": sha256(args.shared_start),
        "source_adapter_sha256": sha256(args.adapter),
        "design_checkpoint_sha256": sha256(args.design_checkpoint),
        "trained_adapter": str(args.output_root / "final_adapter.pt"),
        "trained_adapter_sha256": sha256(args.output_root / "final_adapter.pt"),
        "oracle_sequence_evaluations_per_target": calls,
        "sequence_filter_policy": "ACGT-only and fixed length; GC and homopolymer rejection disabled",
        "diagnostic_only_criteria": "GC 0.30-0.70 and maximum homopolymer length 6",
        "evaluation_boundary": "Top-256 locked before Evaluation, FullMPRA, motif, or sequence diagnostics",
        "search_summary": search_summary,
        "candidates": candidates,
    }
    write_json(lock_path, lock)
    write_json(lock_dir / "manifest.json", {
        "protocol": lock["protocol"],
        "algorithm": args.algorithm,
        "seed": args.seed,
        "target": args.target,
        "candidate_file": str(lock_path),
        "candidate_sha256": sha256(lock_path),
        "oracle_sequence_evaluations": calls,
        "selection_changed_posthoc": False,
    })
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "complete").touch()
    (args.output_root / "formal_search.complete").touch()
    write_json(args.output_root / "progress.json", {"stage": "formal_search_complete", "search_summary": search_summary})
    print(f"LORA_{args.algorithm.upper()}_FORMAL_SEARCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
