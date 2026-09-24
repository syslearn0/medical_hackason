import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import crossval
import fewshot
import predict

POOL = [
    {"user_id": "U001", "date": "2026-04-01", "record": "食事は全量摂取。夜間は良眠。", "scores": {k: 3 for k in predict.FIELDS}},
    {"user_id": "U001", "date": "2026-04-02", "record": "食事は全量摂取。夜間は良眠。", "scores": {k: 3 for k in predict.FIELDS}},
    {"user_id": "U002", "date": "2026-04-01", "record": "食事は全量摂取。入浴は自立。", "scores": {k: 5 for k in predict.FIELDS}},
    {"user_id": "U003", "date": "2026-04-01", "record": "体操参加を拒否。", "scores": {k: 1 for k in predict.FIELDS}},
]


class FewShotTests(unittest.TestCase):
    def test_target_user_excluded(self):
        row = {"user_id": "U001", "date": "2026-04-01", "record": "食事は全量摂取。夜間は良眠。"}
        chosen = fewshot.select_examples(row, POOL, k=10)
        self.assertTrue(chosen)
        self.assertTrue(all(e["user_id"] != "U001" for e in chosen))

    def test_most_similar_first_and_deterministic(self):
        row = {"user_id": "U009", "date": "2026-05-01", "record": "食事は全量摂取。入浴は自立。"}
        first = fewshot.select_examples(row, POOL, k=2)
        self.assertEqual(first[0]["user_id"], "U002")
        self.assertEqual(first, fewshot.select_examples(row, POOL, k=2))

    def test_same_row_never_used_even_if_allowed(self):
        row = {"user_id": "U001", "date": "2026-04-01", "record": "食事は全量摂取。夜間は良眠。"}
        chosen = fewshot.select_examples(row, POOL, k=10, allow_same_user=True)
        self.assertNotIn(("U001", "2026-04-01"), [(e["user_id"], e["date"]) for e in chosen])

    def test_leak_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.json"
            path.write_text(json.dumps({"examples_used": {"U001 2026-04-01": ["U001 2026-04-02"]}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                crossval.check_no_leak(path)

    def test_past_scores_stable_when_new_day_added(self):
        """翌日の記録が入力に追加されても、過去日は再採点されずキャッシュのまま。"""
        valid = {"scores": {k: {"score": None, "evidence": [], "needs_review": True} for k in predict.FIELDS}, "review_note": "情報不足"}
        response = {"done": True, "message": {"content": json.dumps(valid)}}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, prompt, out, examples = root / "in.csv", root / "p.txt", root / "out", root / "ex.csv"
            prompt.write_text("test", encoding="utf-8")
            examples.write_text("利用者ID,日付,記録内容," + ",".join(fewshot.TRUTH_FIELDS) + "\nU002,2026-04-01,完食," + ",".join(["3"] * 8) + "\n", encoding="utf-8")
            argv = ["--input", str(source), "--prompt", str(prompt), "--output", str(out), "--examples", str(examples), "--retries", "0"]
            model = {"name": "m", "digest": "x"}
            source.write_text("user_id,date,record\nU001,2026-04-01,食事不明\n", encoding="utf-8")
            with patch.object(predict, "get_model", return_value=model), patch.object(predict, "request_json", return_value=response) as api:
                self.assertEqual(fewshot.main(argv), 0)
                self.assertEqual(api.call_count, 1)
            source.write_text("user_id,date,record\nU001,2026-04-01,食事不明\nU001,2026-04-02,記録不足\n", encoding="utf-8")
            with patch.object(predict, "get_model", return_value=model), patch.object(predict, "request_json", return_value=response) as api:
                self.assertEqual(fewshot.main(argv), 0)
                self.assertEqual(api.call_count, 1)
            self.assertEqual(json.loads((out / "run.json").read_text(encoding="utf-8"))["cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
