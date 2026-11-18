# Repository and release maintenance

Repository: [ChangzhiJiang/ToDNA](https://github.com/ChangzhiJiang/ToDNA). The repository root contains the core `src/`, selected `scripts/`, configurations, per-run numerical data, generated tables, tests, and documentation.

For subsequent releases:

1. Add verified paper/citation details and a project license when selected by the owner. Preserve third-party notices; do not infer a project-wide license from a dependency's license.
2. Upload the source and numerical results, then check the README and `python3 reproduce.py` from a fresh clone. `.gitignore` keeps credentials, environments, checkpoints, and raw datasets out of routine additions.
3. Keep the release-gap statements visible. Uploading source alone does not make project-specific weights and split manifests publicly downloadable.
4. Keep the repository link in `docs/REVIEWER2_REPLY.md` current. Verify access as a reader before sending that reply.
5. Tag the release used for the rebuttal. If files change, regenerate `SHA256SUMS` or document that the original package manifest applies to the original release. When numerical inputs change, also update their provenance and `INPUT_SHA256.json` deliberately.

The original prepared source package is preserved separately; publication edits update the repository documentation and release checksums without changing the archived original source files.
