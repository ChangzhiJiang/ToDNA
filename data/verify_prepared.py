"""Verify prepared partitions and optionally compare with a frozen reference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

PARTITIONS = ("design_train", "design_validation", "evaluation_train",
              "evaluation_validation", "sealed_test")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify(directory, reference=None):
    manifest = json.loads((directory/"complete.json").read_text())
    files = {part: part+".csv" for part in PARTITIONS}
    files.update(lora_design_train="lora/train.jsonl",
                 lora_design_validation="lora/validation.jsonl")
    counts = {}
    for key, name in files.items():
        path = directory/name
        if digest(path) != manifest["partitions"][key]["sha256"]:
            raise ValueError(f"Manifest checksum mismatch: {name}")
        with path.open() as handle:
            count = sum(1 for _ in handle) - (0 if name.endswith("jsonl") else 1)
        if count != manifest["partitions"][key]["n"]:
            raise ValueError(f"Manifest count mismatch: {name}")
        counts[key] = count
    if sum(counts[p] for p in PARTITIONS) != manifest["eligible"]:
        raise ValueError("Partition counts do not cover eligible records")
    for part, name in (("design_train", "train"), ("design_validation", "validation")):
        with (directory/(part+".csv")).open() as csv_file, \
             (directory/"lora"/(name+".jsonl")).open() as jsonl_file:
            for row, line in zip(csv.DictReader(csv_file), jsonl_file):
                if row["sequence"] != json.loads(line)["sequence"]:
                    raise ValueError("LoRA sequence order differs from design partition")
        if counts[part] != counts["lora_"+part]:
            raise ValueError("LoRA record count differs from design partition")
    membership_counts = {part: 0 for part in PARTITIONS}
    component_partition = {}
    with (directory/"membership.csv").open() as handle:
        for row in csv.DictReader(handle):
            group, part = row["component"], row["partition"]
            if component_partition.setdefault(group, part) != part:
                raise ValueError("Component crosses partitions")
            membership_counts[part] += 1
    if membership_counts != {part: counts[part] for part in PARTITIONS}:
        raise ValueError("Membership counts differ from partitions")
    checked = []
    if reference:
        old = json.loads((reference/"complete.json").read_text())
        for field in ("protocol", "family", "seed", "excluded", "eligible",
                      "components", "largest_component", "partitions"):
            if manifest[field] != old[field]:
                raise ValueError(f"Frozen manifest differs: {field}")
        if sorted(manifest["source_hashes"].values()) != sorted(old["source_hashes"].values()):
            raise ValueError("Source hashes differ from frozen inputs")
        for name in [*files.values(), "sequences.txt", "components.txt", "membership.csv"]:
            if digest(directory/name) != digest(reference/name):
                raise ValueError(f"Frozen output differs byte-for-byte: {name}")
            checked.append(name)
    return {"family": manifest["family"], "eligible": manifest["eligible"],
            "counts": counts, "frozen_files_identical": checked,
            "status": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.reference_dir), indent=2))
