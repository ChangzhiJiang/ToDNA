# Release gaps and known limitations

The repository is useful for inspecting the actual algorithm and reproducing the supplied numerical tables. The following points prevent claiming full fresh-run reproduction of every result:

1. **Weights and sequence assets are external.** Base/adapted models, Design/Evaluation checkpoints, ordered pools/starts, logit caches, and row-level split manifests are not included. No working public download location for these project-specific assets has been added.
2. **One grouped-split helper is missing.** The original preparation script calls `strict_hamming_components`; its C++ source is absent from the available local snapshot. A read-only search of the accessible backup did not locate it. It has not been replaced by an approximate splitter.
3. **Historical software/hardware records are incomplete.** The import-based requirements are not a version-locked environment. Original adaptation GPU model is not identified. Current device information cannot retrospectively prove earlier execution hardware.
4. **Source snapshot is not an exact-run attestation.** Files copied from the original project have verified byte identities relative to that snapshot. Those hashes do not prove that every historical run used precisely that version. The portable launcher and numerical analysis are new packaging additions and are identified separately.
5. **Historical comparison flaws remain visible.** The full factorial is not uniformly identity-matched. One ranked/random pair is excluded for cache-adapter inconsistency. GPA/SVDD budgets and postselection differ. Three seeds are not twelve independent replicates; six fixed-pool search seeds are not six fresh model trainings.
6. **No GPU rerun was performed during packaging.** Static compilation and CPU numerical tests do not validate CUDA kernels, model loading, full training, search trajectories, or checkpoint compatibility.
7. **This is a core release, not all third-party baselines.** It includes ToDNA and its RL code. Nearest-baseline and D3/GPA/SVDD numerical records are included for table reproduction, but the supplementary comparison implementations, complete upstream model implementations, weights, and every revision adapter are not vendored.

Known problematic historical scripts `aggregate_end_to_end_cost.py`, `evaluate_and_aggregate_factorial_3way.py`, and `aggregate_random_start_formal.py` are not used as public result-reproduction entry points. This repository's analysis uses the separately audited numerical records and identity checks. Preserving raw core code does not imply that every branch has been audited or is error-free.

For a later complete rerun release, add the missing helper, immutable data/checkpoint manifests with download locations, matching original environment locks, and a documented fresh-run validation. Do not fill these gaps with invented versions or unverified hardware claims.
