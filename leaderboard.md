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