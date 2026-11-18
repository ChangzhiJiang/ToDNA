from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


COLUMNS = (
    "IDs", "chr", "data_project", "OL", "class",
    "K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC",
    "K562_lfcSE", "HepG2_lfcSE", "SKNSH_lfcSE", "sequence",
)


def main() -> None:
    script = Path(__file__).with_name("prepare_gosai_mpra.py")
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "input.tsv"
        rows = []
        for index in range(7):
            rows.append({
                "IDs": f"id-{index}", "chr": "1", "data_project": "test",
                "OL": "1", "class": "test", "K562_log2FC": str(index + 0.1),
                "HepG2_log2FC": str(index + 0.2), "SKNSH_log2FC": str(index + 0.3),
                "K562_lfcSE": "0.1", "HepG2_lfcSE": "0.1", "SKNSH_lfcSE": "0.1",
                "sequence": "ACGT" * 5,
            })
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        outputs = []
        for name in ("first", "second"):
            output = root / name
            subprocess.run(
                [sys.executable, str(script), str(source), "--output-dir", str(output), "--seed", "17", "--sequence-length", "20"],
                check=True, capture_output=True, text=True,
            )
            outputs.append(output)

        first_manifest = (outputs[0] / "gosai_split_manifest.tsv").read_text()
        second_manifest = (outputs[1] / "gosai_split_manifest.tsv").read_text()
        assert first_manifest == second_manifest
        metadata = json.loads((outputs[0] / "gosai_split_metadata.json").read_text())
        assert metadata["row_count"] == 7
        assert metadata["selected_sequence_length"] == 20
        assert metadata["design_row_count"] == 3
        assert metadata["evaluation_row_count"] == 4
        assert metadata["sequence_length_counts"] == {"20": 7}

    print("Gosai MPRA split test passed")


if __name__ == "__main__":
    main()
