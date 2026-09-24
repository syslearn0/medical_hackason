import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
import export_submission as export
import predict


class SubmissionTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "input.csv"
        source.write_text("user_id,date,record\nu,2026-04-01,記録A\nu,2026-04-02,記録B\n", encoding="utf-8")
        records = predict.read_records(source)
        results = [{"scores": {f: {"score": value, "evidence": [], "needs_review": True} for f in predict.FIELDS}, "review_note": "情報不足"} for value in (2, 3)]
        predict.export_results(root, list(zip(records, results)), [])
        predict.save_json(root / "run.json", {"input": str(source), "requested_records": 2, "completed_records": 2, "failed_records": 0})

    def test_mean_rounding_review_and_full_mae(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            rows, flags, counts, periods = export.aggregate(root, 1, 2, 2)
            self.assertEqual(rows[0]["食事"], "2.50")
            self.assertTrue(flags[0]["食事"])
            self.assertEqual(counts[0]["要確認日数"], 2)
            truth = root / "truth.csv"
            predict.save_csv(truth, ["user_id", *predict.FIELDS], [{"user_id": "u", **{f: "2.00" for f in predict.FIELDS}}])
            result, errors = export.score_submission(rows, truth, periods)
            self.assertEqual(result["values"], 8)
            self.assertEqual(result["mae"], "0.50")

    def test_missing_daily_value_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            path = root / "daily_scores.csv"
            rows = export.read_csv(path, ["user_id", "date", *predict.FIELDS])
            rows[0]["食事"] = ""
            predict.save_csv(path, ["user_id", "date", *predict.FIELDS], rows)
            with self.assertRaises((ValueError, ArithmeticError)):
                export.aggregate(root, 1, 2, 2)

    def test_partial_or_failed_run_not_exported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            run_path = root / "run.json"
            run = json.loads(run_path.read_text())
            run["failed_records"] = 1
            predict.save_json(run_path, run)
            with self.assertRaises(ValueError):
                export.aggregate(root, 1, 2, 2)

    def test_extra_truth_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            rows, _, _, periods = export.aggregate(root, 1, 2, 2)
            truth = root / "truth.csv"
            predict.save_csv(truth, ["user_id", *predict.FIELDS], [{"user_id": u, **{f: 3 for f in predict.FIELDS}} for u in ("u", "v")])
            with self.assertRaises(ValueError):
                export.score_submission(rows, truth, periods)


if __name__ == "__main__":
    unittest.main()
