"""Evaluate saved daily predictions; never send ground truth to Ollama."""
import argparse
import csv
import sys
from pathlib import Path
from predict import FIELDS

TRUTH_FIELDS = ["食事_score", "入浴_score", "運動_score", "排泄_score", "睡眠_score", "認知_意欲_score", "介助負担_score", "リスク_score"]


def indexed(path, id_col, date_col, columns):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = [id_col, date_col, *columns]
        if not all(c in (reader.fieldnames or []) for c in required):
            raise ValueError(f"必要な列がありません: {path}")
        result = {}
        for row in reader:
            key = (row[id_col], row[date_col])
            if not all(key) or key in result:
                raise ValueError(f"空または重複したID・日付: {key}")
            for c in columns:
                if row[c] not in (None, ""):
                    value = float(row[c])
                    minimum = 0 if c in ("介助負担", "リスク", "介助負担_score", "リスク_score") else 1
                    if not value.is_integer() or not minimum <= value <= 5:
                        raise ValueError(f"日次スコアの範囲・型が不正: {key} {c}")
            result[key] = row
    return result


def main():
    p = argparse.ArgumentParser(description="保存済み日次スコアの答え合わせ（未知データへの精度検証ではありません）")
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--truth", required=True, type=Path)
    args = p.parse_args()
    pred = indexed(args.predictions, "user_id", "date", FIELDS)
    truth = indexed(args.truth, "利用者ID", "日付", TRUTH_FIELDS)
    extra = set(pred) - set(truth)
    if extra:
        raise ValueError(f"正解に存在しない予測ID・日付: {sorted(extra)}")
    print(f"正解 {len(truth)}件 / 出力 {len(pred)}件 / 未出力 {len(set(truth) - set(pred))}件")
    print("項目\t採点済み/正解あり\tMAE\t完全一致率")
    for field, label in zip(FIELDS, TRUTH_FIELDS):
        eligible = [k for k in truth if truth[k][label] not in (None, "")]
        diffs = [abs(float(pred[k][field]) - float(truth[k][label])) for k in eligible if k in pred and pred[k][field] not in (None, "")]
        if diffs:
            print(f"{field}\t{len(diffs)}/{len(eligible)}\t{sum(diffs)/len(diffs):.3f}\t{sum(d == 0 for d in diffs)/len(diffs):.1%}")
        else:
            print(f"{field}\t0/{len(eligible)}\tn/a\tn/a")
    print("欠損は0点に置換していません。MAE・一致率は値が出た項目のみ。必ず採点率も確認してください。")
    print("このプロンプトは公開3名を参考に作っています。同じ3名での成績は開発用です。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(1)
