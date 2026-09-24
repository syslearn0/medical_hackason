#!/usr/bin/env python3
"""Leave-one-user-out evaluation of saved daily predictions, broken down per held-out user."""
import argparse
import json
import sys
from pathlib import Path
from predict import FIELDS, ROOT
from evaluate import TRUTH_FIELDS, indexed


def check_no_leak(run_path):
    """fewshot.py の run.json を読み、対象本人の記録が例題に入っていないことを確かめる。"""
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if "examples_used" not in run:
        return "run.json に例題記録がありません（few-shotなしの実行）。プロンプト自体は3名から作られている点に注意。"
    if run.get("allow_same_user"):
        raise ValueError("--allow-same-user で実行した結果は利用者別の交差検証になりません。")
    for target, examples in run["examples_used"].items():
        uid = target.split()[0]
        if any(e.split()[0] == uid for e in examples):
            raise ValueError(f"{target}: 本人の記録が例題に含まれています。")
    return "例題の漏れなし（全件で対象本人以外から例題を選択）"


def summarize(keys, pred, truth):
    rows = []
    for field, label in zip(FIELDS, TRUTH_FIELDS):
        eligible = [k for k in keys if truth[k][label] not in (None, "")]
        diffs = [abs(float(pred[k][field]) - float(truth[k][label]))
                 for k in eligible if k in pred and pred[k][field] not in (None, "")]
        rows.append((field, len(diffs), len(eligible),
                     sum(diffs) / len(diffs) if diffs else None,
                     sum(d == 0 for d in diffs) / len(diffs) if diffs else None))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="利用者別の交差検証（1人ずつ除外して評価）")
    p.add_argument("--predictions", type=Path, default=ROOT / "results/fewshot/daily_scores.csv")
    p.add_argument("--truth", type=Path, default=ROOT / "data/reference/care_hackathon_ground_truth_records_3users.csv")
    args = p.parse_args(argv)
    run_path = args.predictions.parent / "run.json"
    if run_path.exists():
        print(check_no_leak(run_path))
    pred = indexed(args.predictions, "user_id", "date", FIELDS)
    truth = indexed(args.truth, "利用者ID", "日付", TRUTH_FIELDS)
    extra = set(pred) - set(truth)
    if extra:
        raise ValueError(f"正解に存在しない予測ID・日付: {sorted(extra)}")
    users = sorted({uid for uid, _ in truth})
    groups = [(u, [k for k in truth if k[0] == u]) for u in users] + [("全体", list(truth))]
    for name, keys in groups:
        print(f"\n== {name}（{len(keys)}日） ==")
        print("項目\t採点済み/正解あり\tMAE\t完全一致率")
        for field, done, total, mae, exact in summarize(keys, pred, truth):
            print(f"{field}\t{done}/{total}\t" + ("n/a\tn/a" if mae is None else f"{mae:.3f}\t{exact:.1%}"))
    print("\n空欄（null）は0点に置換していません。採点率とあわせて読んでください。")
    print("注意: prompts/scoring.txt 自体は3名全員の正解から作られているため、その分は楽観的な見積もりです。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(1)
