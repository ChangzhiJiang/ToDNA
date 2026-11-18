#!/usr/bin/env python3
"""Standard causal PPO/GRPO with fixed SFT reference and fresh final sampling.

Only Design assets are accepted. This runner never loads Evaluation or test data.
Uses native Vortex cached sampling and one causal training pass per completion.
Experimental backend: requires real-model parity and gradient validation before release.
"""
import argparse
import json
import os
import time
from pathlib import Path
import torch
import src.autoregressive_cached as ar_core
from src.autoregressive_cached import CachedCausalPolicy, parity_metrics, enable_fp32_compute
from evo2 import Evo2
from src.autoregressive_rl import (
    CausalPolicy, gae, group_advantages, policy_loss, value_loss, exact_kl,
)
from src.lora import (
    materialize_autograd_tensors, load_lora_adapter, lora_parameters,
    adapter_state, restore_adapter_state, set_lora_training, save_lora_adapter,
)
from formal_common import (
    disable_flash_attention, sha256, write_json, one_hot_sequences, TASKS, RANGES,
)
from promoter_matched_common import load_promoter_checkpoint, padded_scores
from promoter_matched_common import TASKS as PROMOTER_TASKS, RANGES as PROMOTER_RANGES
from src.predictor import load_activity_checkpoint
from src.profiling import CostProfiler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True, help='strict_independent_v1/assets')
    p.add_argument('--family', choices=['enhancer', 'promoter'], required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--method', choices=['ppo', 'grpo'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--enformer-state-dict', type=Path)
    p.add_argument('--rounds', type=int, default=60)
    p.add_argument('--groups', type=int, default=64)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--final-samples', type=int, default=768)
    p.add_argument('--top-k', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--kl-beta', type=float, default=0.05)
    p.add_argument('--rollout-batch-size', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if min(a.rounds, a.groups, a.epochs, a.final_samples, a.top_k) < 1 or a.group_size < 2:
        p.error('positive counts and group-size >= 2 required')
    if a.top_k > a.final_samples:
        p.error('top-k exceeds final sample budget')
    if a.rollout_batch_size < 1:
        p.error('rollout-batch-size must be positive')
    if not a.smoke and (a.top_k != 256 or a.final_samples != 768):
        p.error('formal runs require exactly 768 final draws and top-256 locks')
    if a.output.exists():
        raise FileExistsError(f'Refusing to overwrite run: {a.output}')
    family = a.assets / a.family
    oracle = family / 'design/oracle.ckpt'
    marker = json.loads((family / 'design/complete.json').read_text())
    if marker['role'] != 'design' or marker['family'] != a.family:
        raise ValueError('Design checkpoint role mismatch')
    if marker['checkpoint_sha256'] != sha256(oracle):
        raise ValueError('Design checkpoint digest mismatch')
    data = a.assets.parent / 'data' / a.family
    if marker['split_manifest_sha256'] != sha256(data / 'complete.json'):
        raise ValueError('Split manifest digest mismatch')
    for partition in ('design_train', 'design_validation'):
        if marker['data_hashes'][partition] != sha256(data / (partition+'.csv')):
            raise ValueError('Design partition digest mismatch')
    adapter = family / 'lora/best_adapter.pt'
    if not (family / 'lora/complete').exists():
        raise RuntimeError('SFT adapter incomplete')
    tasks, ranges = (TASKS, RANGES) if a.family == 'enhancer' else (PROMOTER_TASKS, PROMOTER_RANGES)
    target = tasks.index(a.task)
    length = 200 if a.family == 'enhancer' else 250
    a.output.mkdir(parents=True)
    profiler = CostProfiler(method=a.method.upper(), protocol='strict_independent_autoregressive_cached_v3')
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(a).items()
    }
    metadata = {**arguments, 'assets': str(a.assets), 'output': str(a.output),
                'enformer_state_dict': str(a.enformer_state_dict),
                'protocol': 'strict_independent_autoregressive_cached_v3', 'prefix': 'ACGT',
                'length': length, 'adapter_sha256': sha256(adapter),
                'oracle_sha256': marker['checkpoint_sha256'],
                'evaluation_loaded': False, 'smoke_only': a.smoke,
                'reference': 'fixed initial SFT adapter', 'sampling': 'ACGT softmax, temperature 1',
                'precision': 'FP32 computation; frozen base linear weights retain original storage',
                'source_sha256': {'runner': sha256(Path(__file__)),
                                  'core': sha256(Path(ar_core.__file__))},
                'optimization_reward_queries': a.rounds*a.groups*a.group_size,
                'final_reward_queries': a.final_samples}
    write_json(a.output / 'manifest.json', metadata)
    load_phase = profiler.phase('model_load_and_predictor')
    load_phase.__enter__()
    torch.manual_seed(a.seed)
    evo2_local_path = os.environ.get('EVO2_LOCAL_PATH')
    model = Evo2(
        'evo2_7b',
        local_path=evo2_local_path if evo2_local_path else None,
    ).model
    materialize_autograd_tensors(model)
    config, _ = load_lora_adapter(model, adapter)
    disable_flash_attention(model)
    set_lora_training(model, False)  # deterministic dropout-free old/new/reference policies
    enable_fp32_compute(model)
    device = model.embedding_layer.weight.device
    head = None
    if a.method == 'ppo':
        head = torch.nn.Linear(model.embedding_layer.weight.shape[1], 1).to(device)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    policy = CachedCausalPolicy(model, head)
    params = lora_parameters(model) + ([] if head is None else list(head.parameters()))
    for parameter in params:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0)
    reference = {k: v.clone() for k, v in adapter_state(model).items()}
    if a.family == 'enhancer':
        predictor = load_activity_checkpoint(str(oracle), device=device)
        expected = ('K562_log2FC', 'HepG2_log2FC', 'SKNSH_log2FC')
        actual = tuple(predictor.lightning_model.data_params['tasks']['name'])
        if actual != expected:
            raise ValueError(f'Enhancer task order mismatch: {actual}')
    else:
        if a.enformer_state_dict is None:
            raise ValueError('promoter requires --enformer-state-dict')
        predictor, _ = load_promoter_checkpoint(oracle, a.enformer_state_dict, device)
    predictor.eval()
    load_phase.__exit__(None, None, None)
    low = torch.tensor([ranges[t][0] for t in tasks], device=device)
    high = torch.tensor([ranges[t][1] for t in tasks], device=device)
    off = [i for i in range(3) if i != target]
    def decode(tokens):
        return [''.join(map(chr, row)) for row in tokens.cpu().tolist()]
    @torch.no_grad()
    def reward(tokens):
        q = one_hot_sequences(decode(tokens), device)
        raw = predictor(q) if a.family == 'enhancer' else padded_scores(predictor, q)
        normalized = (raw-low)/(high-low)
        return normalized[:, target] - normalized[:, off].mean(-1)
    generator = torch.Generator(device=device).manual_seed(a.seed)
    prefix = torch.tensor([[65, 67, 71, 84]], device=device)
    started = time.time()
    optimization_phase = profiler.phase(
        'optimization',
        rollout_reward_queries=a.rounds*a.groups*a.group_size,
        policy_update_epochs=a.rounds*a.epochs,
    )
    optimization_phase.__enter__()
    for step in range(a.rounds):
        print(json.dumps({'round': step+1, 'stage': 'rollout'}), flush=True)
        rollouts = []
        # Batch generation, but retain single-sequence gradient microbatches.
        total = a.groups*a.group_size
        for offset in range(0, total, a.rollout_batch_size):
            count = min(a.rollout_batch_size, total-offset)
            tokens, old, old_values = policy.sample(prefix.repeat(count, 1), length, generator)
            for row in range(count):
                seq = tokens[row:row+1]
                rollouts.append((seq, old[row:row+1],
                                 None if old_values is None else old_values[row:row+1], reward(seq)))
            progress = {'round': step+1, 'rounds': a.rounds, 'stage': 'rollout',
                        'rollouts_complete': offset+count, 'rollouts_total': total,
                        'elapsed_seconds': time.time()-started}
            write_json(a.output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
        rewards = torch.cat([x[3] for x in rollouts])
        group_adv = group_advantages(rewards, a.group_size)
        current = {k: v.clone() for k, v in adapter_state(model).items()}
        references = []
        print(json.dumps({'round': step+1, 'stage': 'reference'}), flush=True)
        try:
            restore_adapter_state(model, reference)
            with torch.no_grad():
                for tokens, _, _, _ in rollouts:
                    references.append(policy.teacher(tokens, 4)[0].detach())
        finally:
            restore_adapter_state(model, current)
        advantages, returns = [], []
        for i, (tokens, old, old_values, r) in enumerate(rollouts):
            if head is None:
                advantages.append(group_adv[i].expand_as(old))
                returns.append(None)
            else:
                terminal = torch.zeros_like(old)
                terminal[:, -1] = r
                adv, ret = gae(terminal, old_values)
                advantages.append(adv)
                returns.append(ret)
        if head is not None:
            all_adv = torch.cat(advantages)
            advantages = [(v-all_adv.mean())/all_adv.std(unbiased=False).clamp_min(1e-8) for v in advantages]
        for epoch in range(a.epochs):
            print(json.dumps({'round': step+1, 'stage': 'update', 'epoch': epoch+1}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, (tokens, old, old_values, _) in enumerate(rollouts):
                distribution, logp, values = policy.teacher(tokens, 4)
                if epoch == 0:
                    metrics = parity_metrics(old, logp)
                    # Fail closed until measured native-cache equivalence is established.
                    if metrics['max_ratio_error'] > 0.001:
                        raise RuntimeError(f'Cached rollout/training parity failed: {metrics}')
                loss = policy_loss(logp, old, advantages[i])
                loss = loss + a.kl_beta * exact_kl(distribution, references[i])
                if head is not None:
                    loss = loss + 0.5 * value_loss(values, old_values, returns[i])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite RL objective')
                (loss / len(rollouts)).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
        write_json(a.output / 'progress.json', {'round': step+1, 'rounds': a.rounds,
                   'mean_reward': rewards.mean().item(), 'loss': sum(losses)/len(losses),
                   'elapsed_seconds': time.time()-started})
        print(json.dumps({'round': step+1, 'mean_reward': rewards.mean().item()}), flush=True)
    optimization_phase.__exit__(None, None, None)
    changed = any(not torch.equal(current_value, reference[k]) for k, current_value in adapter_state(model).items())
    if not changed:
        raise RuntimeError('No LoRA policy update detected')
    if head is not None and not bool(head.weight.detach().abs().sum() > 0):
        raise RuntimeError('No learned critic update detected')
    save_lora_adapter(model, a.output / 'final_adapter.pt', config, metadata)
    if head is not None:
        torch.save(head.state_dict(), a.output / 'final_value_head.pt')
    # Fresh RNG stream; no training candidates are reused or selected.
    final_generator = torch.Generator(device=device).manual_seed(a.seed + 1000000000)
    records = []
    print(json.dumps({'stage': 'fresh_final_sampling'}), flush=True)
    final_phase = profiler.phase('fresh_final_sampling', final_reward_queries=a.final_samples)
    final_phase.__enter__()
    for offset in range(0, a.final_samples, a.rollout_batch_size):
        count = min(a.rollout_batch_size, a.final_samples-offset)
        tokens, _, _ = policy.sample(prefix.repeat(count, 1), length, final_generator)
        for row in range(count):
            seq = tokens[row:row+1]
            records.append({'sequence': decode(seq)[0], 'design_reward': reward(seq).item()})
    final_phase.__exit__(None, None, None)
    write_json(a.output / 'final_samples.json', {'sequences': records, 'seed': a.seed+1000000000})
    unique = {r['sequence']: r for r in records}
    if len(unique) < a.top_k:
        raise RuntimeError('Insufficient unique final samples; no unbudgeted resampling allowed')
    selected = sorted(unique.values(), key=lambda x: (-x['design_reward'], x['sequence']))[:a.top_k]
    lock = a.output / ('smoke_topk.json' if a.smoke else 'top256.lock.json')
    write_json(lock, {'sequences': selected, 'source': 'fresh final policy samples',
                     'final_adapter_sha256': sha256(a.output / 'final_adapter.pt'),
                     'evaluation_loaded': False})
    write_json(a.output / 'complete.json', {'lock_sha256': sha256(lock), 'smoke_only': a.smoke,
               'elapsed_seconds': time.time()-started, 'policy_updated': changed,
               'cost_audit': profiler.summary()})


if __name__ == '__main__':
    main()
