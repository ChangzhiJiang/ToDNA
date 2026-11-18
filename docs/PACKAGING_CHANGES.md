# Packaging changes

Prepared 2026-11-18. No training or search job was launched.

- Copied selected original core modules, entry points, tests, and configurations without byte changes. `provenance/SOURCE_MANIFEST.json` records their source-relative locations and original hashes.
- Omitted operational SSH/GPU queue scripts, local environments, caches, raw logs, credentials, weights, and sequence datasets.
- Added `scripts/train_family_adapter.py`. It preserves the archived `queue_strict_independent_lora.py` family reader override (200 bp enhancer / 250 bp promoter) and training arguments, but removes server-specific paths and scheduling. It verifies split hashes and rejects an existing output directory before invoking the unchanged trainer. It does not change the optimization algorithm.
- Added `reproduce.py` for seed-level summaries, valid paired differences, and audited resource accounting. Its statistical helpers are now included directly in that standalone file. The old problematic aggregate scripts are not the default route.
- Exported a numerical-only allowlist from the existing cost audit into `data/cost_phase_records.json`; private source paths were removed. Other numerical input files retain their original bytes.
- Added English documentation, validation tests, release-integrity checks, environment capture, input checksums, and a publication checklist.
- Preserved the original MIT notice for the SeqProp-derived port. No project-wide license or author identity was invented.
- Prepared the GitHub publication copy with a repository overview, directory map, clone instructions, and the actual repository link. Updated publication guidance and the Reviewer 2 response draft; regenerated the release checksums. Archived original source files remain unchanged.
- Removed the top-level `analysis` and `experiments` directories at the owner's request. Updated the test imports, documentation, source inventory, and release checksums. The existing numerical inputs and generated tables are retained; removed files remain recoverable from Git history and the original prepared package.
- Subsequently removed the tracked `results` and `third_party` directories at the owner's request. Generated result tables are now ignored by Git. Updated the source inventory and documentation, preserving the owner's README edits and removal of `SHA256SUMS` and `THIRD_PARTY_NOTICES.md`. The verification tool still checks source/input hashes and reports when the optional whole-release manifest is absent.

The source-manifest paths identify archive-relative origins. One former machine-name directory is represented as `archive/` in the public manifest. Original SHA-256 values still describe the unmodified file bytes.
