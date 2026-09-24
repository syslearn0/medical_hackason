# ローカル介護記録スコア推定

作業フォルダ: `/Users/ONOyuto/Desktop/ハッカソン`

CSVを読み、PC内のOllama / Qwenで1人・1日ずつ採点します。修正版では8項目をすべて数値で出し、不確実性を別ファイルの確認フラグとして保存します。Python標準ライブラリだけを使い、追加のpipインストールは不要です。

## VS Codeから使う

1. Ollamaアプリを起動します。
2. VS Codeでこの作業フォルダを開きます。
3. `⌘ Shift P` → `Tasks: Run Task`（タスクの実行）を選びます。
4. `1. Python・Ollama接続確認` を実行します。
5. `4. サンプル42件を採点（修正版）` を実行します。1件だけならタスク3です。
6. 日次の誤差確認はタスク5、3名分の提出形式CSVとMAEはタスク6です。

タスク6は「14日分の単純平均、小数第2位に四捨五入」を明示的に仮定します。正式な期間集計・丸めの指定が確認できたら合わせて変更してください。7名分の本番データには下記の人数チェックを指定します。

## ターミナルから使う

```bash
cd "/Users/ONOyuto/Desktop/ハッカソン"
.venv/bin/python predict.py --check
.venv/bin/python predict.py --output results/sample_v2
```

日次の答え合わせ:

```bash
.venv/bin/python evaluate.py --predictions results/sample_v2/daily_scores.csv --truth data/reference/care_hackathon_ground_truth_records_3users.csv
```

公開3名の提出形式への変換と、全24値でのMAE:

```bash
.venv/bin/python export_submission.py --run-dir results/sample_v2 --method mean --expected-users 3 --expected-days 14 --decimals 2 --truth sample_3users/care_hackathon_2week_summaries_ground_truth_3users.csv
```

当日の入力が同じ日次形式・14日分の場合の例:

```bash
.venv/bin/python predict.py --input data/input/evaluation.csv --output results/evaluation_v2
.venv/bin/python export_submission.py --run-dir results/evaluation_v2 --method mean --expected-users 7 --expected-days 14 --decimals 2
```

期間・丸め方は大会の正式指定に合わせてください。推定時には正解CSVを読みません。`--truth`は変換後の答え合わせにだけ使い、提出値の決定には使いません。

## 保存される結果

| ファイル | 内容 |
|---|---|
| `daily_scores.csv` | 1人・1日につき8項目の整数推定値 |
| `review_flags.csv` | 日次スコアと同じID・日付・8項目。Trueなら要確認 |
| `review.csv` | 要確認の項目・数値・原文根拠・確認理由 |
| `details.json` | 点数、根拠、フラグ、対応表の候補、適用前後の値と方法 |
| `errors.csv` | 通信・形式検査などの失敗 |
| `run.json` | モデル、採点基準の識別情報、件数、失敗・暫定代替件数、時間 |
| `cache/` | 成功した回答。中断・失敗後の再開に使用 |
| `submission.csv` | 変換コマンドで生成。1人1行、IDと8項目の数値のみ |
| `submission_review_flags.csv` | 利用者ごとの8項目。期間内に一度でも要確認ならTrue |
| `submission_review.csv` | 各項目の要確認日数・既定値使用日数・対象日数 |
| `submission_info.json` | 期間集計の仮定・期間・対象人数、指定時は全値のMAE |
| `submission_errors.csv` | 正解を指定した場合のみ。利用者・項目ごとの絶対誤差 |

提出CSVには確認フラグを混ぜません。フラグは確率や校正済みの信頼度ではなく、人が確認する目印です。再採点したら提出形式への変換も実行し直してください。

## 点数を決める仕組み

1. `prompts/scoring.txt` の暫定基準でQwenが全8項目を推定します。JSON Schemaでnullを禁止します。
2. `scoring_policy.py` の対応表に文章全体が一致し、対応する点数が一意ならその点数を適用します。例:「食事は少量のみ」→2、「体操参加を拒否」→1。
3. 対応表で違う点数が複数出た場合は、Qwenの推定を保持して要確認にします。単純に最低値・最大値にはしません。
4. 表にない文章はQwenの推定を使います。未定義の点数境界は要確認として扱います。
5. 原文根拠がない場合だけ暫定既定値3を使い、`neutral_default`として記録します。正解や良好を意味する値ではなく、情報不足を表す代替推定値です。

`details.json` の `audit` には `exact_rule`（対応表）、`llm`（モデル）、`neutral_default`（暫定既定値）を記録します。対応表でモデルの点数を修正した箇所も要確認です。対応表は原文の文単位で照合し、一般的な否定・予定・過去を含む別文を単語一致だけで採点しない設計です。あらゆる文脈を理解できる正規表現ではないので、複雑な記録は確認してください。

情報不足と通信失敗は別扱いです。通信・形式エラーが再試行でも解消しない場合は、対応表と暫定値で日次の数値を残しますが、失敗件数と終了コードでエラーを報告し、提出形式への変換は停止します。同じ採点コマンドで再実行してください。エラー代替値は成功キャッシュに保存しません。

## 元の実験と比較する

- 1回目の `results/sample` は維持します。修正版はデフォルトで `results/sample_v2` に保存します。
- 旧版のrun.jsonがあるフォルダをpredict.pyの出力先にすると停止します。
- 修正前のコードは `backups/before_numeric_v2` に保存しています。
- 同じ版での再実行は結果一覧を更新します。別の実験は `--output results/experiment_03` のように分けてください。
- 同じ出力先で複数の採点処理を同時に実行しないでください。

公開3名の正解例から作った暫定基準です。公式に示されたのはMAEによる順位づけであり、全点数段階の正式な定義は未確認です。この3名への評価は開発用で、未知の7名への精度を保証しません。1回目の欠損除外MAEと、修正版の全値MAEは評価対象が異なるため、直接の改善率として比較しません。

`fewshot.py`、`extract.py`、`rules.py`、`crossval.py` は別の実験用です。今回の対応表補正・代替値の監査・提出変換の一連の動作は `predict.py` → `export_submission.py` で使ってください。共通Schemaを使うfewshot.pyも数値のみの出力になりますが、予測後の対応表補正はpredict.pyの経路で行います。

## 入力・動作確認

- 入力列は `利用者ID,日付,記録内容` または `user_id,date,record`。同一人物・同一日の複数行は事前に統合してください。
- サマリー専用CSVは未対応です。日次記録の欠落を期間平均から自動で除外しません。
- ローカルの `127.0.0.1:11434` のみに接続し、クラウドモデル・リモートモデルを拒否します。
- 既定モデルは `qwen3.5:35b`、同時処理1件、コンテキスト8192、temperature=0、thinking出力なし。
- `Ctrl+C`で中断し、同じコマンドで再開できます。
- プロンプト・入力・モデルdigest・対応表の変更はキャッシュ識別に反映します。
- テスト: `.venv/bin/python -m unittest discover -s tests -v`

公式資料: [Ollama構造化出力](https://docs.ollama.com/capabilities/structured-outputs)、[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)。
