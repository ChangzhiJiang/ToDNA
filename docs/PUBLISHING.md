# Repository and release maintenance

Repository: [ChangzhiJiang/ToDNA](https://github.com/ChangzhiJiang/ToDNA). The repository root contains the core `src/`, selected `scripts/`, configurations, per-run numerical data, tests, and documentation. Result tables are generated locally and are not tracked.

For subsequent releases:

1. Add verified paper/citation details and a project license when selected by the owner. Preserve third-party notices; do not infer a project-wide license from a dependency's license.
2. Upload the source and numerical inputs, then check the README and `python3 reproduce.py` from a fresh clone. `.gitignore` keeps generated tables, credentials, environments, checkpoints, and raw datasets out of routine additions.
3. Keep the release-gap statements visible. Uploading source alone does not make project-specific weights and split manifests publicly downloadable.
4. Keep the repository link in `docs/REVIEWER2_REPLY.md` current. Verify access as a reader before sending that reply.
5. Tag the release used for the rebuttal. Run `python3 -B tools/verify_release.py` to check the retained source and numerical-input hashes. The current repository does not include a whole-release `SHA256SUMS` manifest; the tool reports this explicitly. When numerical inputs change, update their provenance and `INPUT_SHA256.json` deliberately.

The original prepared source package is preserved separately; publication edits update the repository documentation and release checksums without changing the archived original source files.
