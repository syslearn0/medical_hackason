import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import predict


class ValidationTests(unittest.TestCase):
    def valid(self):
        return {"scores": {k: {"score": 3, "evidence": [], "needs_review": True} for k in predict.FIELDS}, "review_note": "全項目の情報不足、暫定既定値"}

    def test_missing_is_not_zero(self):
        data = self.valid()
        item = predict.validate_result(data, "記載なし")["scores"]["食事"]
        self.assertEqual(item["score"], 3)
        self.assertTrue(item["needs_review"])

    def test_null_is_rejected_even_with_review(self):
        data = self.valid()
        data["scores"]["食事"]["score"] = None
        with self.assertRaises(ValueError):
            predict.validate_result(data, "記録")

    def test_empty_review_note_is_reported_without_changing_scores(self):
        data = self.valid()
        data["review_note"] = ""
        fixed = predict.normalize_review_note(data)
        self.assertEqual(fixed["scores"], data["scores"])
        self.assertEqual(data["review_note"], "")
        self.assertIn("理由を出力しませんでした", fixed["review_note"])
        predict.validate_result(fixed, "情報なし")

    def test_evidence_must_be_in_record(self):
        data = self.valid()
        data["scores"]["食事"] = {"score": 5, "evidence": ["全量摂取"], "needs_review": False}
        with self.assertRaises(ValueError):
            predict.validate_result(data, "食事は少量のみ。")

    def test_boolean_is_not_score(self):
        data = self.valid()
        data["scores"]["食事"]["score"] = True
        with self.assertRaises(ValueError):
            predict.validate_result(data, "記録")

    def test_missing_dimension_rejected(self):
        data = self.valid()
        del data["scores"]["リスク"]
        with self.assertRaises(ValueError):
            predict.validate_result(data, "記録")

    def test_csv_quotes_bom_and_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.csv"
            rows = [{"利用者ID": "user_001", "日付": "2026-04-01", "記録内容": "食事,少量\n入浴拒否"}]
            predict.save_csv(path, list(rows[0]), rows)
            self.assertEqual(predict.read_records(path)[0]["record"], rows[0]["記録内容"])
            predict.save_csv(path, list(rows[0]), rows * 2)
            with self.assertRaises(ValueError):
                predict.read_records(path)

    def test_ground_truth_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "truth.csv"
            path.write_text("利用者ID,日付,記録内容,食事_score\nu,2026-04-01,完食,5\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                predict.read_records(path)

    def test_resume_and_failure_export(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, prompt, out = root / "input.csv", root / "prompt.txt", root / "out"
            source.write_text("user_id,date,record\nu,2026-04-01,食事不明\nu,2026-04-02,記録不足\n", encoding="utf-8")
            prompt.write_text("test", encoding="utf-8")
            response = {"done": True, "message": {"content": json.dumps(self.valid())}}
            argv = ["predict.py", "--input", str(source), "--prompt", str(prompt), "--output", str(out), "--retries", "0"]
            with patch("sys.argv", argv), patch.object(predict, "get_model", return_value={"name": "qwen3.5:35b", "digest": "test"}), patch.object(predict, "request_json", side_effect=[response, RuntimeError("temporary")]):
                self.assertEqual(predict.main(), 1)
            with (out / "daily_scores.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
                self.assertEqual(rows[0]["食事"], "3")
                self.assertEqual(len(rows), 2)
            failed_run = json.loads((out / "run.json").read_text())
            self.assertEqual(failed_run["fallback_records"], 1)
            self.assertEqual(failed_run["failed_records"], 1)
            with patch("sys.argv", argv), patch.object(predict, "get_model", return_value={"name": "qwen3.5:35b", "digest": "test"}), patch.object(predict, "request_json", return_value=response) as api:
                self.assertEqual(predict.main(), 0)
                self.assertEqual(api.call_count, 1)
            run = json.loads((out / "run.json").read_text())
            self.assertEqual(run["completed_records"], 2)
            self.assertEqual(run["cache_hits"], 1)

    def test_old_output_is_protected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, prompt = root / "in.csv", root / "prompt.txt"
            source.write_text("user_id,date,record\nu,d,記録\n", encoding="utf-8")
            prompt.write_text("prompt", encoding="utf-8")
            (root / "run.json").write_text("{}", encoding="utf-8")
            with patch("sys.argv", ["predict.py", "--input", str(source), "--prompt", str(prompt), "--output", str(root)]), patch.object(predict, "get_model", return_value={"name": "m"}):
                with self.assertRaises(ValueError):
                    predict.main()


if __name__ == "__main__":
    unittest.main()
