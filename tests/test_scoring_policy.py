import unittest
import predict
import scoring_policy


class PolicyTests(unittest.TestCase):
    def result(self):
        return {"scores": {k: {"score": 3, "evidence": [], "needs_review": True} for k in predict.FIELDS}, "review_note": "暫定値"}

    def test_small_intake_corrected_with_audit(self):
        raw = self.result()
        raw["scores"]["食事"] = {"score": 3, "evidence": ["食事は少量のみ"], "needs_review": False}
        result, audit = scoring_policy.finalize(raw, "食事は少量のみ。")
        self.assertEqual(result["scores"]["食事"]["score"], 2)
        self.assertTrue(result["scores"]["食事"]["needs_review"])
        self.assertEqual(audit["食事"]["original_score"], 3)
        self.assertEqual(raw["scores"]["食事"]["score"], 3)
        predict.validate_result(result, "食事は少量のみ。")

    def test_opportunity_is_not_participation(self):
        record = "午後に活動機会を設けた。体操参加を拒否。"
        result, audit = scoring_policy.finalize(self.result(), record)
        self.assertEqual(result["scores"]["運動・歩行"]["score"], 1)
        self.assertEqual(audit["運動・歩行"]["source"], "exact_rule")

    def test_negation_history_plan_not_matched(self):
        for record in ["体操参加を拒否しなかった。", "昨日は体操参加を拒否。", "入浴介助ありの予定。", "『体操参加を拒否』という記載を参考にする。", "入浴介助あり？"]:
            self.assertFalse(any(scoring_policy.matching_rules(record).values()), record)

    def test_conflicting_meals_not_arbitrarily_overridden(self):
        raw = self.result()
        raw["scores"]["食事"] = {"score": 4, "evidence": ["朝食は完食", "食事は少量のみ"], "needs_review": False}
        result, audit = scoring_policy.finalize(raw, "朝食は完食。食事は少量のみ。")
        self.assertEqual(result["scores"]["食事"]["score"], 4)
        self.assertTrue(result["scores"]["食事"]["needs_review"])
        self.assertEqual(len(audit["食事"]["rule_candidates"]), 2)

    def test_no_evidence_is_labelled_neutral_fallback(self):
        raw = self.result()
        raw["scores"]["リスク"]["score"] = 0
        result, audit = scoring_policy.finalize(raw, "記載なし")
        self.assertEqual(result["scores"]["リスク"]["score"], 3)
        self.assertEqual(audit["リスク"]["source"], "neutral_default")

    def test_risk_combinations(self):
        cases = [
            ("食事は少量のみ。注意点として、食事低下、高リスクがみられる。", 4),
            ("注意点として、活動低下、高リスクがみられる。服薬拒否があり確認が必要。", 5),
            ("入浴介助あり。経過観察が必要。", 3),
            ("日常動作に介助が必要な場面が多い。", 3),
            ("日常動作に介助が必要な場面が多い。状態は安定。", 2),
            ("入浴介助あり。排泄は自立。", 1),
            ("声かけのみで入浴できた。排泄は自立。", 0),
        ]
        for record, expected in cases:
            result, audit = scoring_policy.finalize(self.result(), record)
            self.assertEqual(result["scores"]["リスク"]["score"], expected, record)
            self.assertEqual(audit["リスク"]["source"], "exact_rule")
            predict.validate_result(result, record)

    def test_risk_not_from_warning_words_alone(self):
        # 「注意」の語だけでは高リスクにしない。列挙がなければ組み合わせで判断する
        for record in ["傾眠や軽度の混乱に注意。", "高リスクではない。", "注意点として、高リスクがみられない。"]:
            self.assertFalse(scoring_policy.matching_rules(record)["リスク"], record)

    def test_api_failure_still_has_numbers_and_review(self):
        result, audit = scoring_policy.finalize(None, "食事は少量のみ。", failure="timeout")
        predict.validate_result(result, "食事は少量のみ。")
        self.assertTrue(all(type(v["score"]) is int and v["needs_review"] for v in result["scores"].values()))
        self.assertEqual(result["scores"]["食事"]["score"], 2)
        self.assertEqual(audit["リスク"]["source"], "neutral_default")


if __name__ == "__main__":
    unittest.main()
