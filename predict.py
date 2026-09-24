#!/usr/bin/env python3
"""Local-only daily care scoring. Python standard library; no pip dependencies."""
import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:11434"
FIELDS = ["食事", "入浴", "運動・歩行", "排泄自立度", "睡眠状態", "認知・意欲", "介助負担", "リスク"]
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 2048}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["scores", "review_note"],
    "properties": {
        "review_note": {"type": "string"},
        "scores": {
            "type": "object", "additionalProperties": False, "required": FIELDS,
            "properties": {
                name: {
                    "type": "object", "additionalProperties": False,
                    "required": ["score", "evidence", "needs_review"],
                    "properties": {
                        "score": {"enum": [None, *range(1 if i < 6 else 0, 6)]},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "needs_review": {"type": "boolean"},
                    },
                } for i, name in enumerate(FIELDS)
            },
        },
    },
}


def request_json(path, payload=None, timeout=15):
    # No proxy, cloud host, environment URL override, or redirects.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollamaへの接続に失敗: {exc}。Ollamaアプリを起動してください。") from exc


def get_model(model):
    if "cloud" in model.lower():
        raise ValueError("クラウドモデルは使用できません。")
    models = request_json("/api/tags").get("models", [])
    selected = next((m for m in models if m.get("name") == model or m.get("model") == model), None)
    if selected is None:
        raise ValueError(f"ローカルモデル {model} がありません。ollama list で確認してください。")
    info = request_json("/api/show", {"model": model})
    if selected.get("remote_host") or info.get("remote_host") or selected.get("remote_model") or info.get("remote_model"):
        raise ValueError("リモートモデルへの接続は拒否しました。ローカルモデルを選んでください。")
    return selected


def read_records(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError("CSVのヘッダーが重複しています。")
        if any("score" in h.lower() or "正解" in h or h in FIELDS for h in headers):
            raise ValueError("正解スコア付きCSVは推定に使えません。data/input の正解なしCSVを指定してください。")
        jp = ("利用者ID", "日付", "記録内容")
        en = ("user_id", "date", "record")
        names = jp if all(k in headers for k in jp) else en
        if not all(k in headers for k in names):
            raise ValueError("日次CSVが必要です。列: 利用者ID,日付,記録内容 または user_id,date,record。サマリー専用CSVは未対応です。")
        rows, seen = [], set()
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"{line}行目: 列数が合いません。")
            if not any((v or "").strip() for v in row.values()):
                continue
            uid, date, record = [(row.get(k) or "").strip() for k in names]
            if not uid or not date or not record:
                raise ValueError(f"{line}行目: ID・日付・記録に空欄があります。")
            if len(record) > 2000:
                raise ValueError(f"{line}行目: 記録が2000文字を超えています。切り捨てず、入力分割とコンテキスト設定を見直してください。")
            key = (uid, date)
            if key in seen:
                raise ValueError(f"{line}行目: 同じ利用者・日付が重複しています: {key}。先に当日の記録を統合してください。")
            seen.add(key)
            rows.append({"user_id": uid, "date": date, "record": record})
    if not rows:
        raise ValueError("CSVにデータがありません。")
    return rows


def validate_result(result, record):
    if not isinstance(result, dict) or set(result) != {"scores", "review_note"}:
        raise ValueError("JSONの最上位キーが不正です。")
    scores = result["scores"]
    if not isinstance(scores, dict) or set(scores) != set(FIELDS):
        raise ValueError("8項目が揃っていません。")
    if not isinstance(result["review_note"], str):
        raise ValueError("review_noteは文字列にしてください。")
    for i, name in enumerate(FIELDS):
        item = scores[name]
        if not isinstance(item, dict) or set(item) != {"score", "evidence", "needs_review"}:
            raise ValueError(f"{name}: 出力形式が不正です。")
        score = item["score"]
        if score is not None and (type(score) is not int or not (1 if i < 6 else 0) <= score <= 5):
            raise ValueError(f"{name}: 点数が暫定範囲外です。")
        if type(item["needs_review"]) is not bool:
            raise ValueError(f"{name}: needs_reviewはbooleanにしてください。")
        evidence = item["evidence"]
        if not isinstance(evidence, list) or any(not isinstance(s, str) or not s.strip() or s not in record for s in evidence):
            raise ValueError(f"{name}: 根拠は記録中に存在する文字列をそのまま引用してください。")
        if score is None and not item["needs_review"]:
            raise ValueError(f"{name}: nullには確認フラグが必要です。")
        if score is not None and not evidence:
            raise ValueError(f"{name}: 点数には根拠が必要です。")
    if any(v["needs_review"] for v in scores.values()) and not result["review_note"].strip():
        raise ValueError("確認事項の理由をreview_noteに記載してください。")
    return result


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_csv(path, columns, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def export_results(out, completed, errors):
    daily, review = [], []
    for row, result in completed:
        daily.append({"user_id": row["user_id"], "date": row["date"], **{k: result["scores"][k]["score"] for k in FIELDS}})
        for k, value in result["scores"].items():
            if value["needs_review"]:
                review.append({"user_id": row["user_id"], "date": row["date"], "項目": k,
                               "推定値": value["score"], "根拠": " / ".join(value["evidence"]), "確認理由": result["review_note"]})
    save_csv(out / "daily_scores.csv", ["user_id", "date", *FIELDS], daily)
    save_csv(out / "review.csv", ["user_id", "date", "項目", "推定値", "根拠", "確認理由"], review)
    save_csv(out / "errors.csv", ["user_id", "date", "error"], errors)
    save_json(out / "details.json", [{**row, **result} for row, result in completed])
    return len(review)


def main():
    p = argparse.ArgumentParser(description="ローカルQwenによる日次採点（暫定基準・提出CSVではありません）")
    p.add_argument("--input", type=Path, default=ROOT / "data/input/care_hackathon_records_3users_14days.csv")
    p.add_argument("--output", type=Path, default=ROOT / "results/sample")
    p.add_argument("--prompt", type=Path, default=ROOT / "prompts/scoring.txt")
    p.add_argument("--model", default="qwen3.5:35b")
    p.add_argument("--limit", type=int, help="先頭N件だけ動作確認")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--retries", type=int, default=1, help="失敗時の追加試行回数")
    p.add_argument("--check", action="store_true", help="Pythonとローカルモデルの接続確認だけ")
    p.add_argument("--validate-only", action="store_true", help="CSV検査のみ。AIは呼び出さない")
    args = p.parse_args()
    if args.retries < 0 or args.timeout < 1 or (args.limit is not None and args.limit < 1):
        p.error("limitとtimeoutは正の整数、retriesは0以上にしてください。")
    if args.check:
        model = get_model(args.model)
        print(f"Python: {sys.version.split()[0]}\n実行環境: {sys.executable}\n接続先: {URL}\nモデル: {model['name']}\n接続確認OK（推論は未実行）")
        return 0
    records = read_records(args.input)
    print(f"CSV検査OK: {len(records)}件 / {len({r['user_id'] for r in records})}名", flush=True)
    if args.validate_only:
        return 0
    if args.limit:
        records = records[:args.limit]
    prompt = args.prompt.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("プロンプトが空です。")
    model = get_model(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "cache"
    cache.mkdir(exist_ok=True)
    completed, errors, cache_hits = [], [], 0
    started = time.monotonic()
    interrupted = False
    config = {"model": args.model, "digest": model.get("digest"), "prompt": prompt, "schema": SCHEMA, "options": OPTIONS, "think": False}
    system = prompt + "\n\n出力JSON Schema:\n" + json.dumps(SCHEMA, ensure_ascii=False)
    try:
        for index, row in enumerate(records, 1):
            print(f"[{index}/{len(records)}] {row['user_id']} {row['date']} ...", flush=True)
            fingerprint = hashlib.sha256(json.dumps({"config": config, "row": row}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            cache_file = cache / f"{fingerprint}.json"
            if cache_file.exists():
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    result = validate_result(cached["result"], row["record"])
                    completed.append((row, result))
                    cache_hits += 1
                    print("  保存済みの結果を再利用", flush=True)
                    export_results(args.output, completed, errors)
                    continue
                except (ValueError, KeyError, TypeError):
                    print("  キャッシュを再作成します", flush=True)
            last_error = ""
            for attempt in range(args.retries + 1):
                try:
                    correction = "" if not last_error else f"\n前回は形式検査に失敗しました。次の点を修正してください: {last_error}"
                    response = request_json("/api/chat", {
                        "model": args.model, "stream": False, "think": False,
                        "format": SCHEMA, "options": OPTIONS, "keep_alive": "5m",
                        "messages": [
                            {"role": "system", "content": system + correction},
                            {"role": "user", "content": json.dumps({"record": row["record"]}, ensure_ascii=False)},
                        ],
                    }, timeout=args.timeout)
                    if response.get("done") is not True or response.get("done_reason") == "length":
                        raise ValueError("出力が完了していません。生成上限やプロンプトを確認してください。")
                    result = validate_result(json.loads(response["message"]["content"]), row["record"])
                    save_json(cache_file, {"result": result, "model": args.model, "digest": model.get("digest"),
                                          "total_duration_ns": response.get("total_duration"), "eval_count": response.get("eval_count")})
                    completed.append((row, result))
                    print("  保存しました", flush=True)
                    break
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    last_error = str(exc)
                    print(f"  試行{attempt + 1}: {last_error}", flush=True)
            else:
                errors.append({"user_id": row["user_id"], "date": row["date"], "error": last_error})
            export_results(args.output, completed, errors)
    except KeyboardInterrupt:
        interrupted = True
        print("\n中断しました。完了済みの結果は保存されます。同じコマンドで再開できます。")
    review_count = export_results(args.output, completed, errors)
    elapsed = round(time.monotonic() - started, 2)
    save_json(args.output / "run.json", {
        "input": str(args.input.resolve()), "model": args.model, "model_digest": model.get("digest"),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "options": OPTIONS, "think": False,
        "requested_records": len(records), "completed_records": len(completed), "failed_records": len(errors),
        "cache_hits": cache_hits, "review_items": review_count, "interrupted": interrupted,
        "elapsed_seconds": elapsed, "output_kind": "daily_provisional_not_submission",
    })
    print(f"完了 {len(completed)}/{len(records)}件、エラー {len(errors)}件、要確認 {review_count}項目、{elapsed}秒\n保存先: {args.output.resolve()}")
    return 130 if interrupted else (1 if errors else 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(1)
