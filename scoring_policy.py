"""Conservative exact-sentence checks derived from the provisional public rubric.

No user IDs, dates, or ground-truth CSVs are used at inference time.
Unknown language is left to the model. Conflicting statements require review.
"""
import copy
import hashlib
import re
from pathlib import Path

VERSION = "numeric-with-review-v2"

# Whole-sentence matches avoid treating negations, plans and past events as facts.
# These examples are provisional, not an official clinical scoring standard.
RULES = {
    "食事": [
        (r"(?:朝食は完食|昼食は全量摂取|食事は全量摂取|主食・副食ともに残さず摂取|食事は十分量を摂取できている)", 5),
        (r"(?:食事は8割程度摂取|少し残したが大部分は摂取|摂取量は概ね良好|主食・副食ともにほぼ食べられていた)", 4),
        (r"(?:食事は半分程度摂取|主食は半量程度、副食は少し残した|摂取量は普段よりやや少なめ|6割程度の摂取にとどまった)", 3),
        (r"(?:食事は少量のみ|声かけしたが摂取量は少なかった|食欲低下あり)", 2),
    ],
    "入浴": [
        (r"(?:声かけのみで入浴できた|浴室内の動作は自立していた|入浴は自立して実施)", 5),
        (r"(?:入浴介助あり|介助下で入浴を実施|移乗と洗体に介助を要した)", 3),
        (r"(?:体調を考慮し部分清拭とした|本日は清拭のみ実施)", 2),
        (r"(?:声かけを行ったが入浴には至らなかった|入浴を拒否した)", 1),
    ],
    "運動・歩行": [
        (r"(?:歩行訓練20分実施|散歩20分を安定して行った)", 5),
        (r"(?:歩行訓練15分を行った|散歩15分実施)", 4),
        (r"(?:歩行訓練10分実施|午後のレクリエーションに参加|レクリエーションに参加|短時間の体操は問題なく実施|上肢体操に参加)", 3),
        (r"(?:散歩は5分程度で終了|短時間のみレクリエーションに参加|声かけにより軽い運動を実施)", 2),
        (r"(?:体操参加を拒否|運動は実施せず|活動への参加はみられなかった|疲労感があり本日の歩行訓練は中止)", 1),
    ],
    "排泄自立度": [
        (r"(?:排泄は自立|排泄は介助なく実施|トイレ動作は自立)", 5),
        (r"(?:排泄は見守りで可能|トイレ誘導後は見守りのみ|声かけと見守りで排泄できた)", 4),
        (r"(?:排泄は一部介助|衣服操作に一部介助を要した|トイレ動作の一部に介助が必要)", 3),
        (r"(?:排泄介助あり|トイレ移乗に介助を要した|排泄動作は介助下で実施)", 2),
    ],
    "睡眠状態": [
        (r"(?:夜間は良眠|夜間覚醒なく休めていた|睡眠状態は良好)", 5),
        (r"(?:一度目覚めたが再入眠できた|軽い中途覚醒あり|夜間覚醒は1回)", 4),
        (r"(?:夜間覚醒は2回|夜間に複数回目覚めた)", 3),
        (r"(?:夜間の眠りが浅く日中の眠気が強い|日中傾眠が目立つ)", 2),
    ],
    "認知・意欲": [
        (r"(?:会話は活発|意欲的に参加|表情明るく反応良好)", 5),
        (r"(?:声かけに対する反応は良好|日中は落ち着いて過ごした|表情は穏やか|レクリエーションに参加)", 4),
        (r"(?:軽度の混乱はあるが指示理解可能|声かけで参加|促しにより短時間参加できた|見守り下で活動に加われた)", 3),
        (r"(?:表情はやや硬い|意欲低下あり|活動への参加は消極的|会話は少なめ)", 2),
    ],
    "介助負担": [
        (r"(?:介助負担は少ない|見守り不要で実施できた)", 0),
        (r"(?:日常動作に介助が必要な場面が多い|複数場面で介助を要した)", 4),
        (r"(?:拒否があり対応に時間を要した|ふらつきがあり常時見守りを要した)", 5),
    ],
    # Risk boundaries are not fixed by the supplied rubric; do not invent a rule.
    "リスク": [],
}


def fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def matching_rules(record):
    sentences = [s.strip() for s in re.split(r"[。\n]", record) if s.strip()]
    return {
        name: [{"score": score, "evidence": s} for pattern, score in patterns
               for s in dict.fromkeys(sentences) if re.fullmatch(pattern, s)]
        for name, patterns in RULES.items()
    }


def finalize(raw_result, record, failure=None):
    """Keep numbers and review status separate; record every correction/fallback."""
    if raw_result is None:
        result = {"scores": {k: {"score": 3, "evidence": [], "needs_review": True} for k in RULES},
                  "review_note": "モデルの採点が得られませんでした。"}
    else:
        result = copy.deepcopy(raw_result)
    matches = matching_rules(record)
    audit, notes = {}, []
    for name, item in result["scores"].items():
        before = item["score"] if raw_result is not None else None
        candidates = matches[name]
        values = {m["score"] for m in candidates}
        source, reason = "llm", ""
        if len(values) == 1:
            score = next(iter(values))
            item["score"] = score
            item["evidence"] = list(dict.fromkeys(m["evidence"] for m in candidates))
            source = "exact_rule"
            if before != score:
                item["needs_review"] = True
                reason = f"暫定対応表を適用: {before} → {score}"
        elif len(values) > 1:
            item["needs_review"] = True
            reason = "異なる点数に対応する記載が複数あるため、モデルの推定値を保持"
        if source != "exact_rule" and (raw_result is None or not item["evidence"]):
            item.update(score=3, evidence=[], needs_review=True)
            source = "neutral_default"
            reason = "根拠付きの推定不能。明示的な暫定既定値3を使用（0点や良好を意味しない）"
        if failure:
            item["needs_review"] = True
            reason = (reason + " / " if reason else "") + "API・形式エラー: " + failure
        if reason:
            notes.append(f"{name}: {reason}")
        audit[name] = {"source": source, "original_score": before, "final_score": item["score"],
                       "reason": reason, "rule_candidates": candidates}
    old_note = result["review_note"].strip()
    result["review_note"] = " / ".join(([old_note] if old_note else []) + notes)
    return result, audit
