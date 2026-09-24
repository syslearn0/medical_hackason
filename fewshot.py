#!/usr/bin/env python3
"""Few-shot daily scoring. Reuses predict.py; examples never come from the target user."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import predict
from predict import FIELDS, OPTIONS, SCHEMA, ROOT
from evaluate import TRUTH_FIELDS


def read_examples(path):
    """正解付きCSVを例題プールとして読む。推定対象の入力には使わない。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = ["利用者ID", "日付", "記録内容", *TRUTH_FIELDS]
        if not all(c in (reader.fieldnames or []) for c in required):
            raise ValueError(f"例題CSVに必要な列がありません: {path}")
        pool = []
        for row in reader:
            scores = {}
            for field, label in zip(FIELDS, TRUTH_FIELDS):
                value = (row[label] or "").strip()
                scores[field] = int(float(value)) if value else None
            pool.append({"user_id": row["利用者ID"].strip(), "date": row["日付"].strip(),
                         "record": row["記録内容"].strip(), "scores": scores})
    if not pool:
        raise ValueError("例題CSVにデータがありません。")
    return pool


def bigrams(text):
    return {text[i:i + 2] for i in range(len(text) - 1)}


def select_examples(row, pool, k, allow_same_user=False):
    """文字bigramのJaccard類似度で上位k件。対象利用者本人の記録は除外（利用者別の交差検証）。"""
    target = bigrams(row["record"])
    candidates = [e for e in pool if allow_same_user or e["user_id"] != row["user_id"]]
    candidates = [e for e in candidates if (e["user_id"], e["date"]) != (row["user_id"], row["date"])]

    def similarity(e):
        other = bigrams(e["record"])
        return len(target & other) / max(1, len(target | other))
    # 類似度が同じなら利用者ID・日付順で決める（毎回同じ例題になるように）
    return sorted(candidates, key=lambda e: (-similarity(e), e["user_id"], e["date"]))[:k]


def format_examples(examples):
    lines = ["【採点例】別の利用者の正解例です。点数の境界の参考にし、例の文言を根拠に引用しないでください。"]
    for i, e in enumerate(examples, 1):
        scores = ", ".join(f"{k}={'null' if v is None else v}" for k, v in e["scores"].items())
        lines.append(f"例{i} 記録: {e['record']}\n例{i} 正解: {scores}")
    return "\n\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="few-shot付き日次採点（例題は対象利用者以外から選ぶ）")
    p.add_argument("--input", type=Path, default=ROOT / "data/input/care_hackathon_records_3users_14days.csv")
    p.add_argument("--examples", type=Path, default=ROOT / "data/reference/care_hackathon_ground_truth_records_3users.csv")
    p.add_argument("--output", type=Path, default=ROOT / "results/fewshot")
    p.add_argument("--prompt", type=Path, default=ROOT / "prompts/scoring.txt")
    p.add_argument("--model", default="qwen3.5:35b")
    p.add_argument("--k", type=int, default=4, help="1件あたりの例題数")
    p.add_argument("--allow-same-user", action="store_true", help="対象本人の別日を例題に使う（交差検証では使わない）")
    p.add_argument("--limit", type=int, help="先頭N件だけ動作確認")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--retries", type=int, default=1)
    args = p.parse_args(argv)
    if args.k < 1 or args.retries < 0 or args.timeout < 1 or (args.limit is not None and args.limit < 1):
        p.error("k・limit・timeoutは正の整数、retriesは0以上にしてください。")
    records = predict.read_records(args.input)
    pool = read_examples(args.examples)
    if args.limit:
        records = records[:args.limit]
    prompt = args.prompt.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("プロンプトが空です。")
    model = predict.get_model(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "cache"
    cache.mkdir(exist_ok=True)
    completed, errors, cache_hits, used = [], [], 0, {}
    started = time.monotonic()
    interrupted = False
    base = prompt + "\n\n出力JSON Schema:\n" + json.dumps(SCHEMA, ensure_ascii=False)
    try:
        for index, row in enumerate(records, 1):
            print(f"[{index}/{len(records)}] {row['user_id']} {row['date']} ...", flush=True)
            examples = select_examples(row, pool, args.k, args.allow_same_user)
            used[f"{row['user_id']} {row['date']}"] = [f"{e['user_id']} {e['date']}" for e in examples]
            system = base + "\n\n" + format_examples(examples)
            config = {"model": args.model, "digest": model.get("digest"), "system": system, "options": OPTIONS, "think": False}
            fingerprint = hashlib.sha256(json.dumps({"config": config, "row": row}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            cache_file = cache / f"{fingerprint}.json"
            if cache_file.exists():
                try:
                    result = predict.validate_result(json.loads(cache_file.read_text(encoding="utf-8"))["result"], row["record"])
                    completed.append((row, result))
                    cache_hits += 1
                    print("  保存済みの結果を再利用", flush=True)
                    continue
                except (ValueError, KeyError, TypeError):
                    print("  キャッシュを再作成します", flush=True)
            last_error = ""
            for attempt in range(args.retries + 1):
                try:
                    correction = "" if not last_error else f"\n前回は形式検査に失敗しました。次の点を修正してください: {last_error}"
                    response = predict.request_json("/api/chat", {
                        "model": args.model, "stream": False, "think": False,
                        "format": SCHEMA, "options": OPTIONS, "keep_alive": "5m",
                        "messages": [
                            {"role": "system", "content": system + correction},
                            {"role": "user", "content": json.dumps({"record": row["record"]}, ensure_ascii=False)},
                        ],
                    }, timeout=args.timeout)
                    if response.get("done") is not True or response.get("done_reason") == "length":
                        raise ValueError("出力が完了していません。生成上限やプロンプトを確認してください。")
                    result = predict.validate_result(json.loads(response["message"]["content"]), row["record"])
                    predict.save_json(cache_file, {"result": result, "model": args.model, "digest": model.get("digest"),
                                                   "examples": used[f"{row['user_id']} {row['date']}"],
                                                   "total_duration_ns": response.get("total_duration")})
                    completed.append((row, result))
                    print("  保存しました", flush=True)
                    break
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    last_error = str(exc)
                    print(f"  試行{attempt + 1}: {last_error}", flush=True)
            else:
                errors.append({"user_id": row["user_id"], "date": row["date"], "error": last_error})
            predict.export_results(args.output, completed, errors)
    except KeyboardInterrupt:
        interrupted = True
        print("\n中断しました。同じコマンドで再開できます。")
    review_count = predict.export_results(args.output, completed, errors)
    elapsed = round(time.monotonic() - started, 2)
    predict.save_json(args.output / "run.json", {
        "input": str(args.input.resolve()), "examples_file": str(args.examples.resolve()),
        "model": args.model, "model_digest": model.get("digest"), "k": args.k, "allow_same_user": args.allow_same_user,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "options": OPTIONS,
        "requested_records": len(records), "completed_records": len(completed), "failed_records": len(errors),
        "cache_hits": cache_hits, "review_items": review_count, "interrupted": interrupted,
        "elapsed_seconds": elapsed, "examples_used": used, "output_kind": "daily_provisional_not_submission",
    })
    print(f"完了 {len(completed)}/{len(records)}件、エラー {len(errors)}件、要確認 {review_count}項目、{elapsed}秒\n保存先: {args.output.resolve()}")
    return 130 if interrupted else (1 if errors else 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(1)
