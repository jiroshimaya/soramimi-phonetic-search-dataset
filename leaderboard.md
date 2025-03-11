# Leaderboard

各手法のRecall@10の評価結果です。

| Method | Recall@10 |
|--------|-----------|
| Mora EditDistance | 0.455 |
| Phoneme EditDistance | 0.672 |
| Vowel Consonant EditDistance | 0.744 |
| KanaSim EditDistance | 0.831 |
| LLM Rerank (gpt-4o-mini) | 0.642 |
| LLM Rerank (gpt-4o) | 0.595 |
| LLM Rerank (gemini-2.0-flash) | 0.565 |
| LLM Rerank (gpt-4.5-preview) | 0.614 |

## 評価方法
- 各手法について、トップ10件の検索結果に対するリコール値を計算
- データセット: soramimi-phonetic-search-dataset v0.0
- パラメータ設定:
  - Vowel Consonant EditDistance: vowel_ratio=0.8
  - KanaSim EditDistance: vowel_ratio=0.8
  - LLM Rerank: Vowel Consonant EditDistance (vowel_ratio=0.5) でtop100を取得（recall@100=0.861）した結果からtop10を再ランク付け 