# Data preprocessing

This directory contains preprocessing programs. Raw datasets, generated
partitions, and the previously bundled numerical summaries are not distributed
in the current repository tree.

## Recovery status

The original `strict_hamming_components` executable was recovered unchanged
from a historical strict-independent experiment archive. It is included as
[`bin/strict_hamming_components_linux_x86_64`](bin/strict_hamming_components_linux_x86_64).
It requires Linux x86-64 and the system C++ runtime; it does not use a GPU.

SHA-256:

```text
df04a3f4b617bad86502f0e0ccf51f672175a72819fa26835f54d19dbef95198
```

**The original `strict_hamming_components.cpp` source remains missing.**
The inspected `DNA_TTO_code.zip` contains its Python caller and test, but no
C++ implementation or executable. No replacement implementation is presented
as original code. Rebuilding this helper from source or running it natively on
macOS/ARM is therefore not currently supported.

## Programs

| File | Purpose |
| --- | --- |
| [`prepare_strict_independent_v1.py`](prepare_strict_independent_v1.py) | Formal component-isolated partitions and LoRA JSONL files |
| [`prepare_gosai_mpra.py`](prepare_gosai_mpra.py) | Earlier row-hash design/evaluation split; **not** the formal component-isolated protocol |
| [`verify_prepared.py`](verify_prepared.py) | Validate counts, hashes, LoRA record order and component membership; optionally compare frozen outputs |
| [`test_strict_hamming_components.py`](test_strict_hamming_components.py) | Independent brute-force oracle for 200/250-bp grouping, RC, boundaries and transitivity |
| [`test_prepare_gosai_mpra.py`](test_prepare_gosai_mpra.py) | Determinism check for the earlier Gosai utility |

The formal Python algorithm preserves the archived eligibility rules, sequence
order, locus grouping, split seed (`20260909`) and component allocation. Release
changes add portable helper/output paths, executable checksum validation,
missing-input checks, and refusal to overwrite nonempty output directories.

## Environment and inputs

Use Linux x86-64 and Python 3.9–3.12 with the preprocessing dependencies:

```bash
python -m pip install -r data/requirements.txt
python data/test_strict_hamming_components.py data/bin/strict_hamming_components_linux_x86_64
python data/test_prepare_gosai_mpra.py
```

`requirements.txt` records the NumPy/Pandas versions used for the verification
below. It is not a historical training-environment lock. The completed runs used
Python 3.9, NumPy 1.26.4 and Pandas 2.2.1 on Linux x86-64.

Place your inputs under an external asset root or the ignored `raw_data/`:

```text
raw_data/
  assets/gosai/Table_S2__MPRA_dataset.txt
  data/ctrl_dna_promoter_matched_v1/
    design_train.csv
    design_validation.csv
    design_test.csv
```

- Enhancer: original Gosai MPRA TSV; required columns are `sequence`, `IDs`,
  `K562_log2FC`, `HepG2_log2FC`, `SKNSH_log2FC`. Source-file SHA-256:
  `f73e8397358a2563098b07cef401b85246e2b84468e573309265198d63b4e834`.
- Promoter: the original three matched-source CSVs, concatenated in
  train/validation/test order; required columns are `sequence`, `JURKAT`,
  `K562`, `THP1`. These input files are distinct from the five partitions
  generated below. They have not been recovered in this release; a raw-data
  download/conversion step for them is not bundled.

## Formal preprocessing

Run from the repository root, using a new empty output directory for each family:

```bash
python data/prepare_strict_independent_v1.py --root raw_data --family enhancer --output-dir outputs/prepared/enhancer
python data/prepare_strict_independent_v1.py --root raw_data --family promoter --output-dir outputs/prepared/promoter
python data/verify_prepared.py outputs/prepared/enhancer
python data/verify_prepared.py outputs/prepared/promoter
```

The promoter command requires the three source files described above. It cannot
run from this checkout alone. The default helper is the recovered executable;
`--helper /path/to/executable` explicitly selects another compatible build.
Changing grouping or component labels can change the frozen split and requires
comparison against the reference data.

Eligible rows have uppercase A/C/G/T sequences of exactly 200 bp (enhancer) or
250 bp (promoter), and finite activity labels. Repeated records are retained.
The helper groups aligned forward/RC pairs with at least 90% identity (at most
20 or 25 substitutions respectively). Connected components, including
transitive links, stay together. Enhancer source-locus variants are then merged.
Indels and shifted homology are not certified by this procedure.

Whole components are allocated to design train/validation, evaluation
train/validation, and sealed test with target proportions 36/4/36/4/20 percent.
The two LoRA files use only the design partitions. Outputs are:

```text
design_train.csv                evaluation_train.csv
design_validation.csv           evaluation_validation.csv
sealed_test.csv                 membership.csv
sequences.txt                   components.txt
lora/train.jsonl                lora/validation.jsonl
complete.json
```

Validate the generated training inputs without starting a GPU job:

```bash
python scripts/train_family_adapter.py --family enhancer --data-root outputs/prepared/enhancer --output-dir outputs/adapter_enhancer --describe-only
```

Use `--family promoter` and the corresponding directories for the 250-bp inputs.
Actual model training requires the separate GPU dependencies and checkpoints.

## Verification on 2026-11-18

| Check | Outcome |
| --- | --- |
| Original executable SHA-256 | Matches recovered archive |
| Brute-force grouping checks, 200 and 250 bp | Passed |
| Enhancer: original full TSV → all prepared outputs | 763,684 eligible records; all 10 data files byte-identical to frozen outputs |
| Promoter: recovered ordered eligible records → all prepared outputs | 17,104 records; all 10 data files byte-identical to frozen outputs |

**The promoter check is a replay of the preprocessing algorithm, not an
end-to-end replay from the three original source CSVs.** Its input order and
labels were recovered from the frozen membership and partition files. The
source hashes consequently differ, and those reconstructed files are not
distributed as originals. Source-level completeness remains blocked by the
missing C++ file and the original promoter input/conversion assets.

The ten compared files comprise five partition CSVs, two LoRA JSONLs,
`sequences.txt`, `components.txt`, and `membership.csv`. `complete.json`
contains machine-dependent source paths and is compared semantically for the
enhancer run. To repeat an exact frozen-input comparison:

```bash
python data/verify_prepared.py outputs/prepared/enhancer --reference-dir /path/to/frozen/enhancer
```

These checks verify data preparation and training-input compatibility; they do
not rerun LoRA training, predictor training or sequence-design experiments.
