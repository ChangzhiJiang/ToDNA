# Source and protocol map

## Adaptation, search, and evaluation

| Stage | Entry point | Important boundary |
| --- | --- | --- |
| Grouped data preparation | `scripts/prepare_strict_independent_v1.py` | Requires original source files and the missing compiled component helper; see release gaps |
| Earlier row-level data preparation | `scripts/prepare_gosai_mpra.py` | Historical utility, not the later grouped split |
| Independent Design/Evaluation predictor fitting | `scripts/train_strict_independent_oracle.py` | Role-specific inputs and checkpoints; no mixing of partitions |
| Earlier predictor variants | `scripts/train_activity_predictor.py`, `scripts/train_full_enformer.py` | Different architectures/configurations; not interchangeable replacements |
| Family LoRA adaptation | `scripts/train_family_adapter.py` calling original `train_evo2_lora_sft.py` | Reads the existing split manifest; retains the historical family-specific input length |
| Candidate sampling | `scripts/sample_initial_sequences.py` | Input generator/adapter, seed and pool identity must match the intended protocol |
| Core search sweep | `scripts/run_search_sweep.py` | Development/general entry; defaults alone do not identify a formal experiment |
| HepG2 canonical/ablation search | `scripts/run_dna_tto_uncapped_hepg2.py` | Explicit arguments, frozen starts, and source hashes identify a run |
| Six-task family-specific search | `scripts/run_dna_tto_uncapped_enhancer.py`, `scripts/run_dna_tto_uncapped_promoter.py` | Enhancers use 200 bp; promoters use 250 bp with the documented predictor padding |
| Factorial contexts and search | `scripts/prepare_factorial_logit_contexts.py`, `scripts/run_dna_tto_factorial.py`, `scripts/run_dna_tto_factorial_group.py` | Same seed does not prove predictor, adapter, pool or cache identity |
| Autoregressive RL | `scripts/run_evo2_autoregressive_rl.py`, `scripts/run_evo2_cached_rl.py`, family-specific `run_evo2_rl_*_seed.py` | Uses the autoregressive implementations, not the historical contextual-edit module |
| Frozen evaluation | `scripts/evaluate_candidates.py`, `scripts/evaluate_full_gosai.py`, `scripts/posthoc_todna_fullmpra.py` | Requires the correct frozen evaluator and candidate lock; never use it to tune search |
| Composition diagnostics | `scripts/build_three_mer_reference.py`, `scripts/compute_paper_sequence_metrics.py` | Reference cohort must match the stated table |

Core source modules and these entry points are preserved without algorithm changes. Commands must run with `src/` on `PYTHONPATH` and with a compatible external model environment.

## Configuration distinctions

| File | Role |
| --- | --- |
| `configs/strict_independent_72_v2.json` | Later six-task comparison, seeds20260920/21/22; same within-family Design/Evaluation assets; protocol specifies 72 task-method-seed searches |
| `configs/dna_tto_uncapped_six_task_v1.json` | Earlier uncapped six-task protocol with different family-specific seeds; do not relabel it as the later three-seed result set |
| `configs/dna_tto_uncapped_hepg2_v1.json` | HepG2 reconstruction/component protocol |
| `configs/evo2_rl_uncapped_table2_v1.json` | Archived RL protocol metadata; consult the associated entry point for the actual backend |
| `configs/locked_search.json` | Earlier development/search configuration; not a universal formal-run default |

Some original entry points expose command-line overrides rather than accepting one configuration file. In particular, the original six-task shell launch used `--edit-penalty-weight 0.0 --prior-weight 0.0 --ranking-prior-penalty 0.0 --disable-sequence-filters`, whereas the Python module defaults are different. The modules were preserved rather than silently changing their historical defaults.

The provided numerical records identify their own protocols. Merely running a current entry point with the same seed is insufficient for exact historical reproduction; checkpoints, pools, start ordering, caches, source versions, and all arguments must also match.

## Supplementary nearest-baseline implementation

`experiments/nearest_baselines/` contains the original revision implementations and tests for ToDNA, FastSeqProp, and Ledidi, plus the relevant protocol configuration. It relies on external Ledidi and on the archived `inputs/` and frozen-context layout. It also depends on the frozen development-selection record. These assets are not shipped; the directory is an inspectable original-source snapshot, not an already populated experiment workspace.

The supplied 108 per-run numerical records cover three methods × six tasks × six search seeds, using a fixed proposal pool. The new `reproduce.py` aggregates those records without loading any of these models.

## Original small tests

The original `scripts/test_prepare_gosai_mpra.py` runs on synthetic records with the standard library. `test_evo2_lora.py`, `test_evo2_sft.py`, `test_rewards.py`, and `test_trust_region.py` require PyTorch. The Hamming-component test additionally requires NumPy/pandas and the original executable.

Original tests use Python `assert`; run them without `-O`. Only the tests explicitly listed as executed in `VALIDATION.md` have been run for this release.
