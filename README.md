# ToDNA

Core implementation and numerical reproduction for **ToDNA**, a framework for predictor-guided DNA sequence optimization with an adapted Evo2 prior.

ToDNA adapts Evo2 to a regulatory-element class using LoRA. At design time, the prior provides candidate sequences, initial sequence logits, and a local reference; a frozen activity predictor guides optimization toward the chosen target-cell preference.

This repository contains the **original ToDNA core implementation**, selected original experiment entry points, archived per-run numerical results, and a standalone table-reproduction program. Source package prepared on 2026-11-18.

It includes LoRA adaptation, frozen-predictor sequence-logit optimization, KL/edit objectives, candidate generation and selection, predictor evaluation, profiling, and PPO/GRPO implementations. Original files are identified by SHA-256 in [`provenance/SOURCE_MANIFEST.json`](provenance/SOURCE_MANIFEST.json).

**Release status:** the numerical tables can be reproduced offline with Python alone. GPU training and search require separately supplied datasets, checkpoints, and a compatible Evo2/Vortex environment. They were not rerun while preparing this release. This core-source release is not a complete, independently verified rerun bundle for every paper experiment. See [release gaps](docs/RELEASE_GAPS.md).

## Repository layout

```text
src/evo2_nabla_dna/   Original adaptation, optimization, predictor, and RL code
scripts/             Selected preparation, training, search, and evaluation entry points
configs/             Experiment configurations
experiments/         Original nearest-baseline comparison implementation
analysis/            Audited numerical aggregation
data/                Archived numerical inputs; no raw sequence datasets
results/             Reproduced CSV and Markdown tables
tests/               Numerical and packaging tests
docs/                Protocols, hardware, data sources, and reproducibility limits
provenance/          Source and numerical-input checksums
```

## Reproduce the numerical tables

Python 3.10 or newer is sufficient; no installation, GPU, model download, network access, or API key is required:

```bash
git clone https://github.com/ChangzhiJiang/ToDNA.git
cd ToDNA
python3 -B reproduce.py
python3 -B -m unittest discover -s tests -v
python3 -B tools/verify_release.py
```

The first command verifies the numerical input hashes and generates ten CSV/Markdown tables in [`results/`](results/). CSV files retain floating-point values; Markdown displays three decimal places. The tests cover accounting rules, duplicate/missing seeds, pairing identities, the known cache-adapter exclusion, and the family-specific adaptation launcher.

The numerical code is in [`analysis/todna_analysis/`](analysis/todna_analysis/). It does not execute training or substitute for the original algorithm.

## Original core code

| Component | Source |
| --- | --- |
| Evo2 forward interface and optimization of sequence logits | [`src/evo2_nabla_dna/dto.py`](src/evo2_nabla_dna/dto.py) |
| Candidate search, scoring, and ranking | [`src/evo2_nabla_dna/design.py`](src/evo2_nabla_dna/design.py) |
| LoRA injection and adapter checkpoint I/O | [`src/evo2_nabla_dna/lora.py`](src/evo2_nabla_dna/lora.py) |
| Continual-pretraining data and language-model loss | [`src/evo2_nabla_dna/sft.py`](src/evo2_nabla_dna/sft.py), [`scripts/train_evo2_lora_sft.py`](scripts/train_evo2_lora_sft.py) |
| Cell-activity rewards and additional objectives | [`src/evo2_nabla_dna/rewards.py`](src/evo2_nabla_dna/rewards.py), `property_rewards.py`, `motif_reward.py` |
| Activity predictors and checkpoint loading | [`src/evo2_nabla_dna/predictor.py`](src/evo2_nabla_dna/predictor.py) |
| Optional sequence constraints | [`src/evo2_nabla_dna/constraints.py`](src/evo2_nabla_dna/constraints.py) |
| Autoregressive PPO/GRPO and cached backend | [`src/evo2_nabla_dna/autoregressive_rl.py`](src/evo2_nabla_dna/autoregressive_rl.py), `autoregressive_cached.py` |
| Historical contextual-edit RL implementation | [`src/evo2_nabla_dna/rl.py`](src/evo2_nabla_dna/rl.py); do not relabel this as autoregressive PPO/GRPO |
| Resource accounting | [`src/evo2_nabla_dna/profiling.py`](src/evo2_nabla_dna/profiling.py) |
| Revision comparison with FastSeqProp/Ledidi | [`experiments/nearest_baselines/`](experiments/nearest_baselines/) |

All original files remain byte-for-byte unchanged. `scripts/train_family_adapter.py` is a new portable wrapper that preserves the 200/250-bp reader selection from the archived family launcher; see [changes](docs/PACKAGING_CHANGES.md). The new statistical program replaces use of the known problematic historical aggregate scripts for the supplied numerical tables.

## Training and experiment entry points

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the entry-point/configuration map and [`docs/DATA_AND_ASSETS.md`](docs/DATA_AND_ASSETS.md) for input sources, selection, and missing assets. The enhancer inputs are 200 bp; the original promoter benchmark uses 250 bp. Do not apply one length to all experiments.

The GPU code imports PyTorch, Evo2/Vortex, and, depending on the entry point, gReLU/Enformer and other packages. [`requirements-core.in`](requirements-core.in) is an import-based dependency inventory, **not a historical version lock**. It is intentionally separate from the zero-dependency numerical workflow.

After preparing a compatible environment:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/train_family_adapter.py --help
python scripts/run_dna_tto_uncapped_hepg2.py --help
python scripts/evaluate_candidates.py --help
python tools/capture_environment.py --output provenance/current_environment.json
```

The environment capture records the current machine's package versions and GPU models; it does not fill in missing historical hardware records. These commands are interfaces, not a claim that the historical assets are bundled.

## Results and interpretation

| Output | What it reproduces |
| --- | --- |
| `results/costs.*` | HepG2, three seeds each: allocated GPU-hours, cumulative runtime, and maximum allocated memory per device |
| `results/six_task_summary.*` | Archived six-task ToDNA/Ctrl-DNA/PPO/GRPO numerical summaries |
| `results/nearest_baselines.*` | Six tasks, three methods, six search seeds on a fixed pool |
| `results/factorial_descriptive.*` | Eight Base/LoRA configurations, three seeds each; descriptive only because the full provenance is not uniformly matched |
| `results/reference_only.*` | Identity-verified Base-versus-LoRA reference pairs, including paired differences |
| `results/start_selection_pairs.*` | Two verified ranked-versus-random pairs, with the excluded third pair explicitly reported |
| `results/regularizers.*` | Archived KL/Edit/None Design-reward and edit summaries |
| `results/retained_before_after.*` | Before/after measurements conditional on retained trajectories |
| `results/d3_pairs.*` | Archived D3 proposal-only versus D3+ToDNA paired numerical results |
| `results/gpa_svdd_cohorts.*` | Full-population versus Evaluation-selected subsets, reported separately |

Use [`docs/STATISTICS.md`](docs/STATISTICS.md) before combining tables. Scores from different predictors, normalization schemes, or selection protocols are not interchangeable. An archived numerical result does not establish that this source snapshot was the exact version used for that run.

## Reproducibility information for reviewers

- [Hardware and compute accounting](docs/HARDWARE_AND_COST.md)
- [LoRA sequence sources, eligibility, and partitioning](docs/DATA_AND_ASSETS.md)
- [Known limitations and assets still needed for a fresh rerun](docs/RELEASE_GAPS.md)
- [Validation performed for this package](docs/VALIDATION.md)
- [Short Reviewer 2 response draft](docs/REVIEWER2_REPLY.md)

Repository: [ChangzhiJiang/ToDNA](https://github.com/ChangzhiJiang/ToDNA). The linked documentation separates the supplied source and numerical summaries from assets still required for fresh GPU runs.

## Licensing and publication

The project owner must select the license for the original ToDNA code before describing it as an open-source release. No new license has been assigned on the owner's behalf. Third-party attribution for the FastSeqProp port is preserved in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Third-party checkpoints and datasets are not redistributed here.

See [`docs/PUBLISHING.md`](docs/PUBLISHING.md) for release maintenance. This package excludes credentials, SSH configuration, server queues, private server logs, model weights, and raw sequence datasets.
