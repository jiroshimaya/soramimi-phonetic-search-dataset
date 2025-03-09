# Leaderboard

各手法のRecall@10の評価結果です。

| Method | Recall@10 |
|--------|-----------|
| Mora EditDistance | 0.455 |
| Phoneme EditDistance | 0.672 |
| Vowel Consonant EditDistance | 0.744 |
| KanaSim EditDistance | 0.831 |

## 評価方法
- 各手法について、トップ10件の検索結果に対するリコール値を計算
- データセット: soramimi-phonetic-search-dataset v0.0.3
- パラメータ設定:
  - Vowel Consonant EditDistance: vowel_ratio=0.8
  - KanaSim EditDistance: vowel_ratio=0.8 