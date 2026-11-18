# ToDNA

Core implementation for **ToDNA**, Reward-Guided Test-Time Optimization for Regulatory DNA Design.

ToDNA adapts Evo2 to a regulatory-element class using LoRA. At design time, the prior provides candidate sequences, initial sequence logits, and a local reference; a frozen activity predictor guides optimization toward the chosen target-cell preference.

This repository contains the **original ToDNA core implementation** and the
retained formal experiment entry points. Shared functions have been extracted
from retired launchers without changing their computation.

## Repository layout

```text
src/                 Original adaptation, optimization, predictor, and RL code
scripts/             17 formal workflow entry points and 3 shared helper modules
configs/             Experiment configurations
reproduce.py         Standalone numerical aggregation and table reproduction
data/                Preprocessing code, validation tools, and recovered grouping executable
```

## Data preprocessing

See [`data/README.md`](data/README.md) for input layouts, dependencies, commands,
and the exact scope of the verification. The recovered grouping executable is
for Linux x86-64. 

```bash
git clone https://github.com/ChangzhiJiang/ToDNA.git
cd ToDNA
python3 -B tools/verify_release.py
```

## Optional numerical aggregation

```bash
python3 -B reproduce.py --data-dir /path/to/archived_numerical_records
```

## Original core code

| Component | Source |
| --- | --- |
| Evo2 forward interface and optimization of sequence logits | [`src/dto.py`](src/dto.py) |
| Candidate search, scoring, and ranking | [`src/design.py`](src/design.py) |
| LoRA injection and adapter checkpoint I/O | [`src/lora.py`](src/lora.py) |
| Continual-pretraining data and language-model loss | [`src/sft.py`](src/sft.py), [`scripts/train_evo2_lora_sft.py`](scripts/train_evo2_lora_sft.py) |
| Cell-activity rewards and additional objectives | [`src/rewards.py`](src/rewards.py), `property_rewards.py`, `motif_reward.py` |
| Activity predictors and checkpoint loading | [`src/predictor.py`](src/predictor.py) |
| Optional sequence constraints | [`src/constraints.py`](src/constraints.py) |
| Autoregressive PPO/GRPO and cached backend | [`src/autoregressive_rl.py`](src/autoregressive_rl.py), `autoregressive_cached.py` |
| Resource accounting | [`src/profiling.py`](src/profiling.py) |

`scripts/train_family_adapter.py` is a portable wrapper that preserves the 200/250-bp reader selection from the archived family launcher.

## Formal experiment entry points

The enhancer inputs are 200 bp; the original promoter benchmark uses 250 bp.

| Stage | Entry points | Inputs and role |
| --- | --- | --- |
| Independent predictor training | [`train_strict_independent_oracle.py`](scripts/train_strict_independent_oracle.py) | Frozen role-specific train/validation partitions; selects the Design or Evaluation checkpoint using its own validation set |
| Family LoRA training | [`train_family_adapter.py`](scripts/train_family_adapter.py), [`train_evo2_lora_sft.py`](scripts/train_evo2_lora_sft.py) | The family wrapper validates split hashes and selects the 200/250-bp reader before calling the trainer |
| Candidate generation | [`sample_initial_sequences.py`](scripts/sample_initial_sequences.py) | Base/LoRA model, sampling seed and budget; saves the fixed pool |
| HepG2 canonical and component ablations | [`run_dna_tto_uncapped_hepg2.py`](scripts/run_dna_tto_uncapped_hepg2.py) | Explicit pool, starts, adapter, Design checkpoint and seed; variants `full`, `no_trust`, `no_lora` |
| Six-task TTO | [`run_dna_tto_uncapped_enhancer.py`](scripts/run_dna_tto_uncapped_enhancer.py), [`run_dna_tto_uncapped_promoter.py`](scripts/run_dna_tto_uncapped_promoter.py) | Shared family-specific pools and starts; uncapped search protocol |
| Three-factor ablation | [`prepare_factorial_logit_contexts.py`](scripts/prepare_factorial_logit_contexts.py), [`run_dna_tto_factorial.py`](scripts/run_dna_tto_factorial.py), [`run_dna_tto_factorial_group.py`](scripts/run_dna_tto_factorial_group.py) | Cache Base/LoRA logits, then run individual cells or four cells sharing one loaded model |
| Autoregressive PPO/GRPO | [`run_evo2_autoregressive_rl.py`](scripts/run_evo2_autoregressive_rl.py), [`run_evo2_cached_rl.py`](scripts/run_evo2_cached_rl.py) | Fixed SFT reference and fresh final sampling; prefix-recompute and cached backends, respectively |
| Frozen predictor evaluation | [`evaluate_candidates.py`](scripts/evaluate_candidates.py), [`evaluate_full_gosai.py`](scripts/evaluate_full_gosai.py), [`posthoc_todna_fullmpra.py`](scripts/posthoc_todna_fullmpra.py) | Score locked candidates with the specified Evaluation or FullMPRA checkpoint |
| Sequence diagnostics | [`build_three_mer_reference.py`](scripts/build_three_mer_reference.py), [`compute_paper_sequence_metrics.py`](scripts/compute_paper_sequence_metrics.py) | Design-pool 3-mer reference versus real-MPRA reference metrics; preserve the intended reference cohort |

Three helper modules support these entry points and are not standalone launchers:

- [`formal_common.py`](scripts/formal_common.py): unchanged search-configuration
  construction, enhancer pool/start validation and reward normalization, DNA
  encoding, and autoregressive attention setup.
- [`promoter_matched_common.py`](scripts/promoter_matched_common.py): promoter
  pool/start validation, checkpoint loading, cell order and predictor padding.
- [`dna_tto_no_filter.py`](scripts/dna_tto_no_filter.py): candidate validation and
  deduplication with GC/homopolymer filtering disabled.

The retained runners no longer import retired experiment launchers. Removed
files include the four `scripts/test_*.py` checks, earlier predictor-training
variants, development searches, the capped matched-search launcher, and the
contextual-edit PPO/GRPO launchers. Their prior versions remain in Git history.
The old development lock and contextual-edit RL configuration were also removed.
Preprocessing validation tools remain in `data/`.

The ordinary and cached autoregressive runners are both retained because the
archived formal queues reference them. They are distinct backends, and a run's
manifest determines which was used. The cached runner retains its original
real-model parity/gradient validation gate; this repository cleanup does not
establish new GPU numerical equivalence between the backends. The contextual-edit
implementation in `src/rl.py` has no retained experiment entry point and is not
the implementation used by these autoregressive runners.

### Protocol and asset selection

The configuration files describe different experiment cohorts:

- [`strict_independent_comparison_v1.json`](configs/strict_independent_comparison_v1.json)
  defines role-isolated data and training. Its status fields are archived protocol
  metadata, not live experiment progress.
- [`strict_independent_72_v2.json`](configs/strict_independent_72_v2.json) specifies
  the later six-task comparison with seeds 20260920/21/22.
- [`dna_tto_uncapped_hepg2_v1.json`](configs/dna_tto_uncapped_hepg2_v1.json) and
  [`dna_tto_uncapped_six_task_v1.json`](configs/dna_tto_uncapped_six_task_v1.json)
  document earlier uncapped cohorts with their own seeds. Do not relabel them
  as the later comparison.

These JSON files are protocol records; the runners generally accept explicit
arguments rather than a universal `--config` option. Match the seed, predictor,
adapter, pool, start order and logit cache to the intended frozen run.
Generate or supply the validated pool/start files before launching search;
this checkout is not a populated experiment workspace.

The six-task enhancer and promoter runners require these explicit flags in
addition to their asset paths and seed:

```text
--edit-penalty-weight 0.0 --prior-weight 0.0 --ranking-prior-penalty 0.0 --disable-sequence-filters
```

Their historical CLI defaults were not silently changed. The code checks the
formal budget and rejects incompatible settings. Evaluation and FullMPRA scoring
belong after candidate locking, not during search or parameter selection.

`train_strict_independent_oracle.py --root ASSET_ROOT` expects prepared data at
`ASSET_ROOT/outputs/strict_independent_v1/data/FAMILY` and the original pretrained
checkpoint layout. Use the preprocessing script's default output layout for this
runner, or place already verified outputs at those paths. The family adapter
wrapper accepts its prepared family directory through `--data-root`.

### Environment and checks

The code imports PyTorch, Evo2/Vortex, and, depending on the entry point, gReLU/Enformer and other packages. [`requirements-core.in`](requirements-core.in) is an import-based dependency inventory, **not a historical version lock**. It is intentionally separate from the zero-dependency numerical workflow.

Some Evo2/Vortex imports initialize Triton's CUDA driver before argument parsing.
Those entry points require a visible CUDA device even for `--help`; hiding all
GPUs can produce a `0 active drivers` error without loading any model weights.

After preparing a compatible environment, run these commands from the repository root. The core modules are imported as `src` and `src.<module>`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/train_family_adapter.py --help
python scripts/run_dna_tto_uncapped_hepg2.py --help
python scripts/evaluate_candidates.py --help
python tools/capture_environment.py --output current_environment.json
```

