# ToDNA

Core implementation for **ToDNA**, Predictor-Guided Test-Time Optimization for Regulatory DNA.

ToDNA adapts Evo2 to a regulatory-element class using LoRA. At design time, the prior provides candidate sequences, initial sequence logits, and a local reference; a frozen activity predictor guides optimization toward the chosen target-cell preference.

This repository contains the **original ToDNA core implementation**.

## Repository layout

```text
src/evo2_nabla_dna/   Original adaptation, optimization, predictor, and RL code
scripts/             Selected preparation, training, search, and evaluation entry points
configs/             Experiment configurations
reproduce.py         Standalone numerical aggregation and table reproduction
data/                Archived numerical inputs; no raw sequence datasets
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

`reproduce.py` creates the ten result tables locally in `results/`. Generated outputs are ignored by Git and are not included in the repository.

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

`scripts/train_family_adapter.py` is a new portable wrapper that preserves the 200/250-bp reader selection from the archived family launcher; see [changes](docs/PACKAGING_CHANGES.md). The new statistical program replaces use of the known problematic historical aggregate scripts for the supplied numerical tables.

## Training and experiment entry points

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the entry-point/configuration map and [`docs/DATA_AND_ASSETS.md`](docs/DATA_AND_ASSETS.md) for input sources, selection, and missing assets. The enhancer inputs are 200 bp; the original promoter benchmark uses 250 bp.

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

## Licensing and publication

The project owner must select the license for the original ToDNA code before describing it as an open-source release. No new license has been assigned on the owner's behalf. External dependencies retain their own licensing conditions. Third-party checkpoints and datasets are not redistributed here.
See [`docs/PUBLISHING.md`](docs/PUBLISHING.md) for release maintenance. This package excludes credentials, SSH configuration, server queues, private server logs, model weights, and raw sequence datasets.
