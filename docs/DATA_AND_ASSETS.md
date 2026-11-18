# Data sources and adaptation-sequence selection

## Historical benchmark sources

The historical project documentation identifies these sources:

| Element class | Source | Eligible length | Design training records | Design validation records |
| --- | --- | ---: | ---: | ---: |
| Enhancer | Gosai et al., Supplementary Table 2, DOI `10.1038/s41586-024-08070-z` | 200 bp | 274,926 | 30,548 |
| Promoter | Human promoter data distributed with the Ctrl-DNA benchmark | 250 bp | 6,157 | 685 |

These are historically reported partition counts, not counts independently reconstructed during packaging. The original row-level split manifests are not included. The enhancer source filename is `Table_S2__MPRA_dataset.txt`; the promoter preparation script expects the benchmark's `design_train.csv`, `design_validation.csv`, and `design_test.csv` source files. Their presence alone does not establish that they match the original bytes.

## How sequences were selected

`scripts/prepare_strict_independent_v1.py` documents and implements the family-specific preparation logic:

1. Retain uppercase A/C/G/T sequences at the eligible family length and with finite required activity labels.
2. Keep exact sequences, reverse complements, and position-aligned equal-length sequence pairs meeting the historical identity threshold in indivisible components. Enhancer variants with the same recorded genomic source position are also grouped. This is not an indel-aware or universal homology guarantee.
3. Allocate whole groups to Design train/validation, Evaluation train/validation, and sealed test. Activity magnitude does not determine allocation.
4. Export only Design training and validation sequences to the adaptation corpus. There is no filter for high target-cell activity.
5. Select the LoRA checkpoint by validation full-vocabulary language-model loss, not by Evaluation-model performance.

The grouped preparation calls an external `strict_hamming_components` helper. Its C++ source is not present in this local release snapshot; this dependency is listed as a release gap rather than replaced with a different splitting algorithm.

`prepare_gosai_mpra.py` is an earlier deterministic row-level split utility. It is retained as original code and tested on synthetic records, but **it is not a replacement for the later grouped split**. Confusing these two preparations would change the benchmark.

The original training function defaults to 200-bp inputs. The historical family launcher overrides that reader to 250 bp for promoters. The portable `train_family_adapter.py` retains this override and validates the pre-existing `complete.json` split hashes before loading a model.

## Assets needed for fresh model execution

| Asset | Included? | Required information |
| --- | --- | --- |
| Evo2 base checkpoint and compatible Vortex code | No | Exact checkpoint revision and upstream software versions |
| Family-specific LoRA adapters | No | Adapter bytes, SHA-256, training manifest, checkpoint-selection record |
| Design and Evaluation predictor checkpoints | No | Separate model identities, input conventions, task/channel order |
| Enformer pretrained state and FullMPRA checkpoint | No | Exact source/version and checksum |
| Source datasets and grouped split membership | No | Raw source bytes, split manifest, component assignments |
| Frozen candidate pools, ordered starts, and logit caches | No | Content hashes and identity matches within each paired comparison |
| Frozen per-run numerical summaries | Yes | `data/`, protected by `provenance/INPUT_SHA256.json` |

Numerical summaries are sufficient for the supplied table reproduction, but not for retraining, rescoring sequences, or checking every biological preprocessing step. Exact sequence/weight release locations should be added when the authors make those assets available.

The Design/Evaluation split limits downstream data overlap. Exposure during generic foundation-model pretraining remains unknown. The supplied results are computational predictions, not new wet-lab measurements.
