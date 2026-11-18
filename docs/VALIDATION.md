# Validation performed

Validation date: 2026-11-18. Environment details are in `provenance/validation_environment.json`.

## Executed

- Ran `python3 -B reproduce.py`: ten tables regenerated successfully from the supplied immutable numerical inputs.
- Ran the 12 tests in `tests/test_analysis.py`: all passed. Coverage includes sample SD, missing/nonfinite measurements, duplicate/missing seeds, paired identity failures, the real start-cache exclusion, cost aggregation, archived cost regression, reference-only paired differences, full report generation, and the family-launcher input/hash checks.
- Ran original `scripts/test_prepare_gosai_mpra.py` with synthetic rows: passed. This checks the earlier deterministic row-split utility, not the missing grouped-split executable.
- Compiled every supplied Python source in memory without importing external models or writing bytecode.
- Parsed all supplied JSON files and checked local Markdown links.
- Verified preserved original-file hashes and numerical input hashes.
- Scanned the package for recognizable API-key/private-key formats, private account paths, and internal IP addresses: no matches. This is a pattern scan, not a guarantee that every conceivable secret format is detectable.
- Extracted the final ZIP into a separate temporary directory, reran the numerical workflow, tests, and integrity checks there, and verified that generated tables did not change the release checksums.

## Not executed

GPU training/search, model downloads, CUDA kernel checks, predictor loading, external API calls, and full baseline reruns. PyTorch is unavailable in the local validation interpreter, so the original PyTorch-dependent tests were not executed. There is no claim of end-to-end GPU reproducibility or validation of all historical scientific conclusions.

The package's portability claim is limited to source inspection and its executed numerical workflow. Fresh GPU runs still require the assets and environment described in `RELEASE_GAPS.md`.
