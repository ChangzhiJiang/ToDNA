#!/usr/bin/env python3
"""Check release hashes, source preservation, syntax, links, and accidental secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", "__pycache__", ".venv", "venv"}


def files():
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and not IGNORED_PARTS.intersection(p.relative_to(ROOT).parts)
                  and p.relative_to(ROOT).parts[0] != "results"
                  and p.name != ".DS_Store"
                  and p.relative_to(ROOT).as_posix() != "provenance/current_environment.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect():
    issues = []
    python_count = 0
    patterns = {
        "private_key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        "google_api_key": r"AIza[0-9A-Za-z_-]{30,}",
        "wandb_api_key": r"wandb_v1_[0-9A-Za-z_]{30,}",
        "huggingface_token": r"hf_[A-Za-z0-9]{25,}",
        "github_token": r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
        "private_account_path": r"/(?:Users|home|mnt/nas)/[A-Za-z0-9_.-]+/",
        "internal_ipv4": r"\b10\.(?:\d{1,3}\.){2}\d{1,3}\b",
    }
    for path in files():
        rel = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            issues.append(f"Unexpected symlink: {rel}")
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            issues.append(f"Unexpected binary file: {rel}")
            continue
        for label, pattern in patterns.items():
            if re.search(pattern, text):
                issues.append(f"Potential {label}: {rel}")
        if path.suffix == ".py":
            compile(text, rel, "exec")
            python_count += 1
        if path.suffix == ".json":
            json.loads(text)
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#", 1)[0]
                if target and not (path.parent/target).exists():
                    issues.append(f"Broken local link: {rel} -> {target}")
    original_count = 0
    for item in json.loads((ROOT/"provenance/SOURCE_MANIFEST.json").read_text()):
        if item["modification"] == "none":
            original_count += 1
            if digest(ROOT/item["path"]) != item["sha256_original"]:
                issues.append(f"Original file changed: {item['path']}")
    for rel, expected in json.loads((ROOT/"provenance/INPUT_SHA256.json").read_text()).items():
        if digest(ROOT/rel) != expected:
            issues.append(f"Numerical input changed: {rel}")
    if issues:
        raise SystemExit("\n".join(issues))
    return {"python_files_compiled": python_count, "original_files_hash_verified": original_count,
            "credential_and_path_pattern_scan": "no matches", "local_markdown_links": "valid"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Explicitly update the whole-release SHA256SUMS")
    args = parser.parse_args()
    result = inspect()
    manifest = ROOT/"SHA256SUMS"
    if args.write_manifest:
        manifest.write_text("".join(f"{digest(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in files() if p != manifest))
        result["whole_release_manifest"] = "written"
    elif manifest.exists():
        for line in manifest.read_text().splitlines():
            expected, rel = line.split("  ", 1)
            if digest(ROOT/rel) != expected:
                raise SystemExit(f"Release hash mismatch: {rel}")
        result["whole_release_manifest"] = "verified"
    else:
        result["whole_release_manifest"] = "not included; source and numerical input hashes verified"
    print(json.dumps(result, indent=2))
