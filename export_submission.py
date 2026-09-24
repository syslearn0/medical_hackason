#!/usr/bin/env python3
"""Explicit period-mean export, separate review flags, and optional full-cell MAE."""
import argparse
import csv
import json
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from predict import FIELDS, ROOT, read_records, save_csv, save_json, validate_result

TRUTH = ["食事_score_平均", "入浴_score_平均", "運動_score_平均", "排泄_score_平均", "睡眠_score_平均", "認知_意欲_score_平均", "介助負担_score_平均", "リスク_score_平均"]


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(required) <= set(columns):
            raise ValueError(f"必要な列がないか重複しています: {path}")
        rows = list(reader)
        if any(None in r for r in rows):
            raise ValueError(f"CSVの列数が合いません: {path}")
    return rows


def unique(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if not all(key) or key in result:
            raise ValueError(f"ID・日付の空欄または重複: {key}")
        result[key] = row
    return result


def aggregate(run_dir, expected_users, expected_days, decimals):
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("interrupted") or run.get("failed_records") or run.get("fallback_records"):
        raise ValueError("未解決の処理エラーまたは中断があります。predict.pyを再実行してください。")
    inputs = unique(read_records(Path(run["input"])), ["user_id", "date"])
    if run.get("requested_records") != len(inputs) or run.get("completed_records") != len(inputs):
        raise ValueError("全件が完了していません。--limitなしで採点してください。")
    predictions = unique(read_csv(run_dir / "daily_scores.csv", ["user_id", "date", *FIELDS]), ["user_id", "date"])
    details = unique(json.loads((run_dir / "details.json").read_text(encoding="utf-8")), ["user_id", "date"])
    if set(inputs) != set(predictions) or set(inputs) != set(details):
        raise ValueError("入力と採点結果のID・日付が一致しません。欠損・余分な行があります。")
    users = sorted({u for u, _ in inputs})
    if len(users) != expected_users:
        raise ValueError(f"人数が一致しません: 期待{expected_users}名、実際{len(users)}名")
    for key, row in details.items():
        if row["record"] != inputs[key]["record"]:
            raise ValueError("入力記録が採点後に変更されています。再採点してください。")
        validate_result({"scores": row["scores"], "review_note": row["review_note"]}, row["record"])
        for field in FIELDS:
            value = Decimal(predictions[key][field])
            if not value.is_finite() or value != Decimal(row["scores"][field]["score"]):
                raise ValueError(f"CSVと根拠ファイルの点数が不一致: {key} {field}")
    output, flags, counts, periods = [], [], [], {}
    unit = Decimal(1).scaleb(-decimals)
    for user in users:
        keys = sorted(k for k in inputs if k[0] == user)
        if len(keys) != expected_days:
            raise ValueError(f"{user}: 期待{expected_days}日、実際{len(keys)}日")
        periods[user] = {"start": min(k[1] for k in keys), "end": max(k[1] for k in keys)}
        out, flag = {"user_id": user}, {"user_id": user}
        for field in FIELDS:
            average = sum(Decimal(predictions[k][field]) for k in keys) / len(keys)
            out[field] = str(average.quantize(unit, rounding=ROUND_HALF_UP))
            review_days = sum(details[k]["scores"][field]["needs_review"] for k in keys)
            default_days = sum(details[k].get("audit", {}).get(field, {}).get("source") == "neutral_default" for k in keys)
            flag[field] = bool(review_days)
            counts.append({"user_id": user, "項目": field, "推定値": out[field],
                           "要確認日数": review_days, "既定値使用日数": default_days, "対象日数": len(keys)})
        output.append(out)
        flags.append(flag)
    return output, flags, counts, periods


def score_submission(rows, truth_path, periods):
    with truth_path.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    id_col, labels = ("user_id", FIELDS) if "user_id" in header else ("利用者ID", TRUTH)
    truth = unique(read_csv(truth_path, [id_col, *labels]), [id_col])
    pred = unique(rows, ["user_id"])
    if set(pred) != set(truth):
        raise ValueError("予測と正解の利用者IDが一致しません。部分評価は行いません。")
    errors = []
    for key, row in pred.items():
        target = truth[key]
        if "期間開始" in target and (target["期間開始"] != periods[key[0]]["start"] or target["期間終了"] != periods[key[0]]["end"]):
            raise ValueError(f"正解と推定の期間が違います: {key[0]}")
        for field, label in zip(FIELDS, labels):
            prediction, correct = Decimal(row[field]), Decimal(target[label])
            if not prediction.is_finite() or not correct.is_finite():
                raise ValueError("非数値・欠損・無限大は採点できません。")
            errors.append({"user_id": key[0], "項目": field, "提出相当値": str(prediction),
                           "正解": str(correct), "絶対誤差": str(abs(prediction - correct))})
    total = sum(Decimal(r["絶対誤差"]) for r in errors)
    return {"users": len(rows), "values": len(errors), "absolute_error_sum": str(total),
            "mae": str(total / len(errors)),
            "mae_by_field": {f: str(sum(Decimal(r["絶対誤差"]) for r in errors if r["項目"] == f) / len(rows)) for f in FIELDS},
            "note": "全値を対象。期間単純平均・指定桁丸めを仮定した公開サンプルの開発評価。未知の利用者での検証ではない。"}, errors


def main(argv=None):
    p = argparse.ArgumentParser(description="日次推定を提出形式へ変換。期間平均を使うという仮定を明示して実行します。")
    p.add_argument("--run-dir", type=Path, default=ROOT / "results/sample_v2")
    p.add_argument("--method", required=True, choices=["mean"])
    p.add_argument("--expected-users", type=int, required=True)
    p.add_argument("--expected-days", type=int, required=True)
    p.add_argument("--decimals", type=int, required=True)
    p.add_argument("--truth", type=Path, help="任意。出力値を決めた後の答え合わせにのみ使う")
    args = p.parse_args(argv)
    if args.expected_users < 1 or args.expected_days < 1 or not 0 <= args.decimals <= 6:
        p.error("人数・日数は正、桁数は0～6にしてください。")
    rows, flags, counts, periods = aggregate(args.run_dir, args.expected_users, args.expected_days, args.decimals)
    score, errors = score_submission(rows, args.truth, periods) if args.truth else (None, None)
    save_csv(args.run_dir / "submission.csv", ["user_id", *FIELDS], rows)
    save_csv(args.run_dir / "submission_review_flags.csv", ["user_id", *FIELDS], flags)
    save_csv(args.run_dir / "submission_review.csv", ["user_id", "項目", "推定値", "要確認日数", "既定値使用日数", "対象日数"], counts)
    save_json(args.run_dir / "submission_info.json", {"method": args.method, "decimals": args.decimals,
              "rounding": "ROUND_HALF_UP", "expected_users": args.expected_users, "expected_days": args.expected_days,
              "periods": periods, "review_items": sum(r["要確認日数"] > 0 for r in counts),
              "aggregation_note": "期間単純平均・丸めの方式は仮定。大会の正式指定と一致するか別途確認が必要。",
              "evaluation": score})
    if errors:
        save_csv(args.run_dir / "submission_errors.csv", ["user_id", "項目", "提出相当値", "正解", "絶対誤差"], errors)
    print(f"提出形式CSV: {args.run_dir / 'submission.csv'}\n{len(rows)}名×8項目、欠損なし。期間単純平均・小数{args.decimals}桁を仮定。")
    if score:
        print(f"MAE = {score['absolute_error_sum']} / {score['values']} = {score['mae']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, ArithmeticError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(1)
