# Hardware and computational accounting

## What is supported by the archived records

| Experiment | Recoverable hardware information | Limitation |
| --- | --- | --- |
| Original family-specific LoRA adaptation | One allocated GPU per training job, according to the archived training description | GPU model not identified in the preserved training log |
| Supplementary KL/Edit/None comparison | Node with four Tesla V100 GPUs, 32 GiB each | Node inventory does not by itself show that each individual run used all four GPUs |
| Cost comparison | Per-phase visible GPU counts in `data/cost_phase_records.json` | Heterogeneous devices/allocations; not a same-device throughput benchmark |
| Missing PPO/GRPO seed completion | Archived completion described as RTX 4090 execution | Does not identify the device for every earlier phase |
| Numerical reproduction in this repository | CPU only; validation environment recorded in `provenance/validation_environment.json` | No GPU training/search validation was performed for the release |

Do not infer a historical GPU model from a current server inventory or a server nickname. Unknown details are left unknown.

## Cost table

The generated `results/costs.csv` has three methods and three matched HepG2 seeds per method: 20260920, 20260921, and 20260922.

- **Cumulative runtime:** sum of audited selected-process elapsed times across stages and seeds. It is neither multi-server wall-clock completion time nor time-to-convergence under identical hardware.
- **Allocated GPU-hours:** audited process elapsed time multiplied by the number of visible/allocated devices, accumulated across the canonical path. This is not observed GPU-busy time.
- **Peak allocated memory per GPU:** maximum single-device PyTorch allocated-memory peak across selected stages and seeds, converted using 1 GiB = 2^30 bytes. This is not a sum across devices, reserved CUDA memory, or total process memory.

The ToDNA path charges model loading for the LoRA search group, LoRA candidate-pool generation, ranking/scoring starts, LoRA initial logits, and the all-LoRA search. For a mixed Base/LoRA logit process, only explicitly timed Base loading/forward phases are removed; unassigned process overhead remains charged. Base-only peaks are excluded. No division by eight factorial cells is applied.

PPO and GRPO use their corresponding archived canonical run records. Existing formal records are retained and missing method/seed pairs are filled from their formal completion records. The duplicate GRPO seed20260922 completion is not counted twice.

All methods have three seeds and 256 retained outputs. The budget of 768 has a method-specific role: initial proposals for ToDNA versus final samples in the RL protocol. Equal numerical counts do not imply identical computation.

Checkpoint pretraining, queue wait, and post-lock independent Evaluation scoring are outside this measured comparison. Candidate/Design scoring and optimization are inside the canonical measured path. Explicit phase times can sum to less than process totals because the latter retain unassigned overhead; do not replace audited process totals by a naive sum of named phases.

The supplied audit reproduces approximately 29.106/119.447/33.430 allocated GPU-hours for ToDNA/PPO/GRPO. The different device allocations mean these measurements should not be presented as intrinsic same-hardware speedup ratios.
