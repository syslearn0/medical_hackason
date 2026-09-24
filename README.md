# ローカル介護記録スコア推定

このフォルダのPythonプログラムがCSVを読み、PC内のOllamaへ1件ずつ送り、Qwenの回答を保存します。VS Codeは編集・実行に使います。

作業場所: `/Users/ONOyuto/Desktop/ハッカソン`

## 最初に使う

1. Ollamaアプリを起動します。
2. VS Codeの「ファイル → フォルダーを開く」で、この「ハッカソン」フォルダを開きます。
3. `⌘ Shift P` でコマンドパレットを開き、`Tasks: Run Task`（タスクの実行）を選びます。
4. `1. Python・Ollama接続確認` を選びます。
5. 続いて `3. 1件だけ採点` を実行します。
6. 左のファイル一覧から `results/demo/daily_scores.csv` を開きます。

タスクは専用Pythonを直接呼び出すため、Python拡張機能のインストールや仮想環境の手動選択は必須ではありません。初回にVS Codeの信頼確認が出た場合は、内容を確認してこの作業フォルダを信頼するとタスクを実行できます。

1件の結果を確認したら、`4. サンプル42件を採点` を実行します。処理中はターミナルに `[1/42]` のように進捗が出ます。続いて `5. サンプル42件を答え合わせ` で誤差を確認できます。

## ターミナルから実行する場合

VS Codeの「ターミナル → 新しいターミナル」を開き、順番に実行します。

```bash
cd "/Users/ONOyuto/Desktop/ハッカソン"
.venv/bin/python predict.py --check
.venv/bin/python predict.py --validate-only
.venv/bin/python predict.py --limit 1 --output results/demo
```

42件すべて:

```bash
.venv/bin/python predict.py --output results/sample
```

答え合わせ:

```bash
.venv/bin/python evaluate.py --predictions results/sample/daily_scores.csv --truth data/reference/care_hackathon_ground_truth_records_3users.csv
```

当日のデータが日次CSVなら、`data/input/evaluation.csv` として置きます。

```bash
.venv/bin/python predict.py --input data/input/evaluation.csv --output results/evaluation
```

英語の入力列 `user_id,date,record` と、公開サンプルの `利用者ID,日付,記録内容` に対応します。人数やIDは固定していません。サマリー専用CSVはこの版では処理しません。

## 出力を読む

| ファイル | 内容 |
|---|---|
| `daily_scores.csv` | 1人・1日ごとの8項目。未判定は空欄 |
| `details.json` | 入力記録、各項目の点数・原文根拠・確認フラグ |
| `review.csv` | 人が確認する項目と理由 |
| `errors.csv` | 通信や出力形式などの処理失敗。失敗ゼロならヘッダーのみ |
| `run.json` | モデル、設定、完了件数、処理時間など |
| `cache/` | 完了済み回答。途中再開用 |

これは日次採点の試作環境です。**`daily_scores.csv` は7名分・1人1行の提出CSVではありません。** 期間の集計方法、整数か小数か、丸め方が確定したら提出用変換を追加します。欠損や要確認を残したまま提出しないでください。

## 採点基準を変える

`prompts/scoring.txt` を編集します。現在の基準は公開3名の例から作った暫定版で、全段階の定義は揃っていません。点数の境界が不明な場合は要確認として記録します。正式な基準が届いたら置き換えます。点数範囲が変わる場合は `predict.py` のSchemaと検査も合わせて変更します。

プロンプト・モデルのdigest・入力・設定が変わるとキャッシュを使わず再採点します。同じ出力フォルダでは結果一覧をその実行対象で更新します。比較実験は `--output results/experiment_02` のように別フォルダへ保存してください。同じ出力フォルダで複数の実行を同時に走らせないでください。

このプロンプトは3名全員を参考にしています。同じ3名の答え合わせは開発用であり、未知の7名での精度を保証しません。利用者別の交差検証を行う場合は、各回の基準作りや例題から評価対象の利用者の正解を除く必要があります。

## 動作と再開

- 接続先は `127.0.0.1:11434` に固定。クラウドモデル・リモートモデルを拒否し、プロキシも使いません。このプログラムから外部APIへデータを送信しません。
- Python標準ライブラリだけを使用します。`pip install` は不要です。
- デフォルトは `qwen3.5:35b`、同時処理1件、コンテキスト8192、temperature=0、thinking出力なし。完全な再現性は保証しません。
- 1回の待ち時間は300秒、失敗時の再試行は1回。必要なら `--timeout 600` を指定します。切り詰められた出力は成功扱いにしません。
- 根拠が原文に存在するか、8項目が揃うか、暫定範囲内かを検査します。形式検査は意味の正しさを保証しません。
- `Ctrl+C` で中断できます。同じコマンドで再開すると完了済み回答を再利用します。
- 1記録2000文字超や、同じ人・同じ日の重複は自動で切り捨てず停止します。当日形式に合わせて統合・分割を検討してください。
- 正解CSVは `data/reference` に分離し、推定時の入力としては拒否します。評価処理だけが読みます。
- `data` と `results` はGitの追跡対象から外しています。元のDownloads内のサンプルは変更しません。

Ollama全体のクラウド機能も無効にしたい場合は、公式の `disable_ollama_cloud` 設定を確認してください。このセットアップでは既存のOllama共通設定は変更しません。

## 困った場合

- 接続失敗: Ollamaアプリを起動し、再度 `--check`。
- 遅い: まず初回のモデル読込完了を待ちます。1件の実測時間を確認してください。必要なら小型のローカルモデルを別途用意し `--model モデル名` で比較できます。
- 全件が終わらない: `errors.csv` と `run.json` を確認し、同じコマンドで再開。
- 値が空欄: 情報不足を意味します。`review.csv` を確認し、0点に置き換えないでください。

## プログラムの検査

```bash
.venv/bin/python -m unittest discover -s tests -v
```

通信を模擬したテストです。根拠の捏造、欠損・0点の区別、正解混入、重複、途中失敗とキャッシュ再開を確認します。モデルの採点精度のテストではありません。

公式資料: [Ollama構造化出力](https://docs.ollama.com/capabilities/structured-outputs)、[Ollama FAQ](https://docs.ollama.com/faq)、[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)。
