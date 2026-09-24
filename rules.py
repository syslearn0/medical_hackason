#!/usr/bin/env python3
"""Facts (from extract.py) -> daily scores -> per-user period averages, evaluated leave-one-user-out."""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from predict import FIELDS, ROOT, save_csv
from evaluate import TRUTH_FIELDS

# 手書きルール: prompts/scoring.txt の暫定基準をそのまま表にしたもの
TABLE = {
    "食事": {"全量": 5, "ほぼ全量": 4, "やや少なめ": 3, "半量": 3, "少量": 2},
    "入浴": {"自立": 5, "声かけのみ": 5, "介助あり": 3, "清拭": 2, "拒否・未実施": 1},
    "運動・歩行": {"積極的参加": 5, "通常参加": 3, "短時間・促しで参加": 2, "不参加・拒否・中止": 1},
    "排泄自立度": {"自立": 5, "見守り": 4, "一部介助": 3, "介助": 2},
    "睡眠状態": {"良眠": 5, "覚醒1回・軽い中途覚醒": 4, "覚醒複数回": 3, "眠りが浅い・日中傾眠": 2},
    "認知・意欲": {"活発・意欲的": 5, "穏やか・反応良好": 4, "促しで参加・軽い混乱": 3, "意欲低下・消極的・表情硬い": 2},
}
# 記載がない日の既定値（中間。平均を大きく歪めないため）
DEFAULT = {"食事": 3, "入浴": 3, "運動・歩行": 2, "排泄自立度": 3, "睡眠状態": 3, "認知・意欲": 3, "介助負担": 2.5, "リスク": 2}


def labels(facts, item):
    return facts[item]["labels"]


def structured(record):
    """形の決まった事実はLLMに頼らず正規表現で読む（9bは分数の読み落としや注意点の捏造があった）。"""
    minutes = [int(m) for m in re.findall(r"(?:散歩|歩行訓練)(\d+)分", record)]
    listed = re.search(r"注意点として、(.+?)がみられる", record)
    return {"運動分数": sum(minutes) if minutes else None,
            "注意点": set(listed.group(1).split("、")) if listed else set(),
            "服薬拒否": "服薬拒否" in record}


def hand_scores(facts, record):
    """1日分の事実から8項目。複数ラベルが付いたら平均（平均で評価されるので小数でよい）。"""
    fixed = structured(record)
    out = {}
    for item, table in TABLE.items():
        values = [table[l] for l in labels(facts, item) if l in table]
        if item == "食事" and not values and "食欲低下" in labels(facts, item):
            values = [2]
        if item == "運動・歩行" and fixed["運動分数"] is not None:
            # 公開例: 5分=2, 10分=3, 15分=4, 20分=5。分数があればラベルより優先
            m = fixed["運動分数"]
            values = [5 if m >= 20 else 4 if m >= 15 else 3 if m >= 10 else 2]
        out[item] = statistics.mean(values) if values else DEFAULT[item]
    overall, notes = set(labels(facts, "全体")), fixed["注意点"]
    assisted = ("介助あり" in labels(facts, "入浴") or "清拭" in labels(facts, "入浴")) + \
               any(l in labels(facts, "排泄自立度") for l in ("一部介助", "介助"))
    if overall & {"拒否で対応に時間", "常時見守り"}:
        out["介助負担"] = 5
    elif "介助が多い" in overall or "介助負担大" in notes:
        out["介助負担"] = 4
    elif "介助負担少ない" in overall:
        out["介助負担"] = 0
    else:
        out["介助負担"] = [1, 3, 3.5][assisted]
    if "高リスク" in notes:
        out["リスク"] = 4 + fixed["服薬拒否"]
    elif "経過観察必要" in overall or notes:
        out["リスク"] = 3
    elif "介助が多い" in overall:
        out["リスク"] = 3 - ("安定" in overall)
    else:
        out["リスク"] = 1 if assisted else 0
    return out


def feature_set(facts):
    s = {f"{k}:{l}" for k in facts if isinstance(facts[k], dict) for l in facts[k]["labels"]}
    if facts["運動分数"] is not None:
        s.add(f"分数:{'20+' if facts['運動分数'] >= 20 else '10+' if facts['運動分数'] >= 10 else '<10'}")
    return s


def learned_scores(facts, record, train, k=3):
    """学習版: 対象以外の利用者の正解から。項目に関係するラベルが同じ日の平均、なければ近傍、なければ手書き。"""
    hand = hand_scores(facts, record)
    target = feature_set(facts)
    out = {}
    for item in FIELDS:
        prefix = [f"{item}:"] + (["分数:"] if item == "運動・歩行" else [])
        if item in ("介助負担", "リスク"):
            # 全体像で決まる項目は、全ラベルの重なりで近い日（上位k件）の平均
            ranked = sorted(train, key=lambda t: -len(target & t["features"]) / max(1, len(target | t["features"])))
            out[item] = statistics.mean(t["truth"][item] for t in ranked[:k])
            continue
        key = frozenset(f for f in target if f.startswith(tuple(prefix)))
        same = [t["truth"][item] for t in train if frozenset(f for f in t["features"] if f.startswith(tuple(prefix))) == key]
        out[item] = statistics.mean(same) if key and same else hand[item]
    return out


def load(facts_path, truth_path):
    rows = json.loads(facts_path.read_text(encoding="utf-8"))
    facts = {(r["user_id"], r["date"]): r["facts"] for r in rows}
    records = {(r["user_id"], r["date"]): r["record"] for r in rows}
    truth = {}
    with truth_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            truth[(row["利用者ID"], row["日付"])] = {k: float(row[t]) for k, t in zip(FIELDS, TRUTH_FIELDS)}
    return facts, records, truth


def average_by_user(daily):
    users = sorted({u for u, _ in daily})
    return {u: {k: statistics.mean(v[k] for (uu, _), v in daily.items() if uu == u) for k in FIELDS} for u in users}


def mae(pred, truth, keys):
    return statistics.mean(abs(pred[k][f] - truth[k][f]) for k in keys for f in FIELDS)


def main(argv=None):
    p = argparse.ArgumentParser(description="抽出事実→ルール採点の利用者別交差検証")
    p.add_argument("--facts", type=Path, default=ROOT / "results/extract_9b/facts.json")
    p.add_argument("--truth", type=Path, default=ROOT / "data/reference/care_hackathon_ground_truth_records_3users.csv")
    p.add_argument("--direct", type=Path, default=ROOT / "results/sample/daily_scores.csv", help="比較用: LLM直接採点の結果")
    p.add_argument("--output", type=Path, default=ROOT / "results/rules")
    args = p.parse_args(argv)
    facts, records, truth = load(args.facts, args.truth)
    missing = set(truth) - set(facts)
    if missing:
        print(f"注意: 抽出結果がない日 {len(missing)}件は評価から除外: {sorted(missing)[:5]}...")
    keys = sorted(set(truth) & set(facts))
    users = sorted({u for u, _ in keys})
    hand = {k: hand_scores(facts[k], records[k]) for k in keys}
    learned = {}
    for u in users:  # 利用者別の交差検証: uを予測するときはu以外の正解だけ使う
        train = [{"features": feature_set(facts[k]), "truth": truth[k]} for k in keys if k[0] != u]
        for k in keys:
            if k[0] == u:
                learned[k] = learned_scores(facts[k], records[k], train)
    methods = {"手書きルール": hand, "学習ルール(交差検証)": learned}
    if args.direct.exists():
        direct = {}
        with args.direct.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                direct[(row["user_id"], row["date"])] = {k: (float(row[k]) if row[k] else None) for k in FIELDS}
        # 空欄は除いて平均（元の実装で平均を取った場合の値）。日次MAEは空欄を既定値で埋めて計算
        methods["LLM直接採点(参考)"] = {k: {f: (v if v is not None else DEFAULT[f]) for f, v in direct[k].items()} for k in keys if k in direct}
        direct_avg = {u: {f: statistics.mean([direct[k][f] for k in keys if k[0] == u and k in direct and direct[k][f] is not None] or [DEFAULT[f]])
                          for f in FIELDS} for u in users}
    true_avg = average_by_user({k: truth[k] for k in keys})
    print(f"評価対象 {len(keys)}日 / {len(users)}名\n")
    print("手法\t日次MAE\t平均値MAE(大会指標)")
    for name, daily in methods.items():
        avg = direct_avg if name.startswith("LLM直接") else average_by_user(daily)
        print(f"{name}\t{mae(daily, truth, [k for k in keys if k in daily]):.3f}\t{mae(avg, true_avg, users):.3f}")
    print("\n項目別の平均値MAE")
    print("項目\t" + "\t".join(methods))
    avgs = {n: (direct_avg if n.startswith("LLM直接") else average_by_user(d)) for n, d in methods.items()}
    for f in FIELDS:
        print(f + "\t" + "\t".join(f"{statistics.mean(abs(avgs[n][u][f] - true_avg[u][f]) for u in users):.3f}" for n in methods))
    args.output.mkdir(parents=True, exist_ok=True)
    for tag, daily in (("hand", hand), ("learned", learned)):
        avg = average_by_user(daily)
        save_csv(args.output / f"averages_{tag}.csv", ["user_id", *FIELDS], [{"user_id": u, **{f: round(avg[u][f], 2) for f in FIELDS}} for u in users])
    print("\n注意: 手書きルールとLLMのプロンプトは3名全員の例から作った基準に基づくため楽観的。学習ルールだけが利用者別の交差検証。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(1)
