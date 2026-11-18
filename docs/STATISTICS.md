# Statistical and comparison conventions

- Summarize seeds within each task/method/condition using arithmetic mean and sample SD (`ddof=1`). Sequence observations nested within a seed are not independent experimental replicates.
- The six-task archive has three seeds. The nearest-baseline comparison has six **search** seeds on a fixed pool. Do not combine them into one population.
- A single observation has an unestimable sample SD, represented as missing, not zero.
- Missing/nonfinite values and duplicate seed records are errors. Expected seed sets are checked before aggregation.
- Paired differences are computed from the within-seed differences. Differences of marginal SDs do not give the SD of the difference.
- Reference-only pairing verifies Design checkpoint, adapter, pool, starts, and logit-cache identities. Both reference arms retain an Evo2 reference; this is not a prior-removal experiment.
- Ranked/random-start pairing additionally checks cache-adapter consistency and budgets. Seed20260920 fails this check and is explicitly excluded; seeds20260921 and 20260922 are retained.
- The full 24-cell factorial is descriptive: mixed predictor/adapter provenance prevents a strictly controlled interpretation of its full main/interaction effects. Recomputing SD cannot repair this confounding. `data/factorial_provenance.csv` records the later seed21/22 checks; an unverified match is not proof that no matching artifact exists anywhere.
- KL/Edit/None summaries are Design-reward outcomes at the archived coefficients, not a matched-fidelity coefficient sweep or independent predictive validation.
- Retained before/after summaries condition on eventual output selection. They are not full-pool averages or causal estimates of the optimizer's general effect.
- D3 paired improvements add TTO compute and retain Evo2-derived initialization/reference. They establish neither equal cost nor full replacement of Evo2.
- GPA/SVDD full 768-sequence outcomes and Evaluation-selected Top-256 outcomes are separate endpoints. Their Design-query budgets differ; the selected subset is not directly comparable with a Design-selected, blind-evaluated cohort.
- Raw MinGap, normalized MinGap, normalized mean margin, and scores from different predictors are separate metrics. See each output caption for its scale.

`data/table2_paired_summary.csv` preserves the earlier Table 2 paired-difference summary for reference. It has rounded summary values rather than raw seed pairs and is not used to manufacture per-seed observations or a new exact uncertainty estimate.

No significance or state-of-the-art claim is produced automatically. The scripts write explicit numerical tables and interpretation limits.
