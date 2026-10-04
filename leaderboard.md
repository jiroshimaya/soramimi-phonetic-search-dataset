# Leaderboard

各手法のRecall@10の評価結果です。

| Method | Recall@10 |
|--------|-----------|
| Mora EditDistance | 0.455 |
| Phoneme EditDistance | 0.672 |
| Vowel Consonant EditDistance | 0.744 |
| KanaSim EditDistance | 0.831 |
| LLM Rerank (gpt-4o-mini) | 0.444 |
| LLM Rerank (gpt-4o) | 0.508 |
| LLM Rerank (gemini-2.0-flash) | 0.496 |
| LLM Rerank (gpt-4.5-preview) | 0.583 |
| LLM Rerank (gpt-5.4) | 0.573 |
| LLM Rerank (gpt-5.4, medium, step-by-step) | 0.936 |

## 評価方法
- 各手法について、トップ10件の検索結果に対するリコール値を計算
- データセット: soramimi-phonetic-search-dataset v0.0
- パラメータ設定:
  - Vowel Consonant EditDistance: vowel_ratio=0.8
  - KanaSim EditDistance: vowel_ratio=0.8
  - LLM Rerank: 以下の手順でリランクを行う
    1. Vowel Consonant EditDistance (vowel_ratio=0.5) で上位100件を取得
    2. 正解が100件に含まれない場合は、下位のものと入れ替え
    3. 順序によるバイアスを避けるため、候補をあいうえお順にソート
    4. LLMに候補を渡して上位10件を選択させる

細かな prompt/input の派生実験は、別リポジトリの `soramimi-phonetic-search-experiments` に移しました。
<!-- gpt6-family-five-variant-results -->


## GPT-6 系モデルの5試行比較

各条件は固定150クエリ・100候補・同じ指示内容の言い換え5通りで評価しています。値は macro Recall@10 の平均 ± 標本標準偏差（n=5、ddof=1）です。完了した150クエリの試行結果を順次保存し、平均と標準偏差は5試行が揃った条件のみ掲載します。未完了の条件は pending と表示します。GPT-6.1 Solはnoneに非対応のためmediumのみ評価します。

評価データの固定コミット: `6072e13bed37dd2f8eb780e61b6154d21cff2e31`。5試行の完了条件: 0/15。詳細結果は [results/gpt6_family](reproduce_leaderboard/results/gpt6_family/) を参照してください。

Recall@10; mean ± sample standard deviation over five wording variants (n = 5).

| Model | Reasoning | Prompt | Easy (65) | Medium (47) | Hard (38) | Overall | Status |
|---|---|---|---:|---:|---:|---:|---|
| gpt-6-sol | medium | step_by_step | — | — | — | — | pending (1/5 variants) |
| gpt-6-luna | medium | step_by_step | — | — | — | — | pending (1/5 variants) |
| gpt-6.1-sol | medium | step_by_step | — | — | — | — | pending (1/5 variants) |
