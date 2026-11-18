import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis"), str(ROOT / "scripts")]
from todna_analysis.statistics import aggregate, cost_totals, pairs, summary
from train_family_adapter import build_plan
import reproduce


class StatisticalTests(unittest.TestCase):
    def test_sample_sd_and_unestimable_singleton(self):
        self.assertEqual(summary([1, 2, 3]), {"n": 3, "mean": 2, "sample_sd": 1})
        self.assertIsNone(summary([1])["sample_sd"])
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                summary(values)

    def test_duplicate_and_missing_seed_fail(self):
        row = {"arm": "a", "seed": 1, "score": 2}
        for records in ([row, row], [row]):
            with self.assertRaises(ValueError):
                aggregate(records, ["arm"], ["score"], [1, 2])

    def test_pair_identity_is_checked(self):
        rows = [{"seed": 1, "arm": "a", "pool": "same"},
                {"seed": 1, "arm": "b", "pool": "different"}]
        accepted, excluded = pairs(rows, "arm", "a", "b", ["pool"], [1])
        self.assertEqual(accepted, [])
        self.assertEqual(excluded, [{"seed": 1, "reason": "pool"}])

    def test_missing_pair_fails(self):
        with self.assertRaises(ValueError):
            pairs([{"seed": 1, "arm": "a"}], "arm", "a", "b", (), [1])

    def test_real_start_exclusion(self):
        accepted, excluded = pairs(reproduce.load("start_pairs.json"), "arm", "random", "ranked",
            ("pool_id", "design_id", "run_adapter_id", "cache_adapter_id", "queries", "retained"), reproduce.SEEDS)
        self.assertEqual([s for s,_,_ in accepted], [20260921, 20260922])
        self.assertEqual([r["seed"] for r in excluded], [20260920])
        self.assertAlmostEqual(accepted[0][2]["final_reward"]-accepted[0][1]["final_reward"], .04223837540484965)

    def test_cost_adds_time_but_takes_peak(self):
        rows = [{"method": method, "seed": seed, "allocated_gpu_hours": 4,
                 "run_elapsed_seconds": 3600, "peak_allocated_bytes": seed * 2**30}
                for method in ("ToDNA", "PPO", "GRPO") for seed in (1, 2, 3)]
        total = cost_totals(rows, [1, 2, 3])[0]
        self.assertEqual(total["total_allocated_gpu_hours"], 12)
        self.assertEqual(total["cumulative_runtime_hours"], 3)
        self.assertEqual(total["peak_allocated_GiB_per_GPU"], 3)
        with self.assertRaises(ValueError):
            cost_totals(rows + [rows[0]], [1, 2, 3])

    def test_archived_cost_regression(self):
        rows = cost_totals(reproduce.load("cost_per_seed.csv"), reproduce.SEEDS)
        self.assertAlmostEqual(rows[0]["total_allocated_gpu_hours"], 29.10627095823073)
        self.assertAlmostEqual(rows[1]["cumulative_runtime_hours"], 37.80787768363713)
        self.assertAlmostEqual(rows[2]["peak_allocated_GiB_per_GPU"], 19.57141351699829)

    def test_actual_reference_hashes_and_paired_difference(self):
        accepted, excluded = pairs(reproduce.load("reference_pairs.json"), "kl_reference", "base", "lora",
            ("design_checkpoint_sha256", "adapter_sha256", "pool_sha256", "starts_sha256", "logit_cache_sha256"), reproduce.SEEDS)
        self.assertFalse(excluded)
        self.assertEqual(len(accepted), 3)
        stats = summary(b["evaluation_min_gap_mean"]-a["evaluation_min_gap_mean"] for _,a,b in accepted)
        self.assertAlmostEqual(stats["mean"], .008646647135416666)

    def test_all_reports_in_isolated_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            reports = reproduce.run(Path(temp))
            self.assertEqual(len(reports), 10)
            self.assertTrue((Path(temp)/"REPORT_INDEX.json").is_file())


class FamilyLauncherTests(unittest.TestCase):
    def make_data(self, root, family):
        (root/"lora").mkdir()
        parts = {}
        for split in ("train", "validation"):
            data = b"synthetic-test-placeholder\n"
            (root/"lora"/(split+".jsonl")).write_bytes(data)
            parts["lora_design_"+split] = {"sha256": hashlib.sha256(data).hexdigest()}
        (root/"complete.json").write_text(json.dumps({"family": family, "partitions": parts}))

    def test_family_length_matches_archived_launcher(self):
        for family, length in (("enhancer", 200), ("promoter", 250)):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.make_data(root, family)
                plan = build_plan(family, root, root/"new_output")
                self.assertEqual(plan["sequence_length"], length)

    def test_wrong_input_hash_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_data(root, "promoter")
            (root/"lora/train.jsonl").write_text("changed")
            with self.assertRaises(ValueError):
                build_plan("promoter", root, root/"new_output")

    def test_family_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_data(root, "enhancer")
            with self.assertRaises(ValueError):
                build_plan("promoter", root, root/"new_output")


if __name__ == "__main__":
    unittest.main()
