# soramimi-phonetic-search-dataset

音韻検索システムの評価用データセット。替え歌の歌詞から構築された特定ジャンルの単語ペアを収録。

[日本語](https://github.com/jiroshimaya/soramimi-phonetic-search-dataset/blob/main/README.md) | [English](https://github.com/jiroshimaya/soramimi-phonetic-search-dataset/blob/main/README.en.md)
## 概要

このデータセットは、音韻が類似した単語を検索するシステムの評価に使用できます。
特定ジャンルの単語だけで歌詞の音韻を模倣する替え歌（いわゆる「〇〇で歌ってみた」）から抽出した単語ペアを含み、特に韻やリズムの一致を重視する音韻検索の評価に適しています。

各手法の評価結果は[leaderboard](https://github.com/jiroshimaya/soramimi-phonetic-search-dataset/blob/main/leaderboard.md)をご覧ください。

## 使い方

### インストール

```bash
pip install soramimi-phonetic-search-dataset
```

### 基本的な使用例

```python
from soramimi_phonetic_search_dataset import evaluate_ranking_function

# カスタムのランキング関数を定義
def my_ranking_function(query_texts: list[str], wordlist_texts: list[str]) -> list[list[str]]:
    # ここにあなたの音韻的類似度に基づくランキングロジックを実装
    return ranked_wordlists

# 評価の実行
recall = evaluate_ranking_function(ranking_func=my_ranking_function, topn=10)
print(f"Recall@10: {recall}")
```

### サンプルのランキング関数

以下のランキング関数が実装済みです：

- `rank_by_mora_editdistance`: モーラ編集距離によるランキング
- `rank_by_vowel_consonant_editdistance`: 母音と子音の編集距離によるランキング
- `rank_by_phoneme_editdistance`: 音素編集距離によるランキング
- `rank_by_kanasim`: [KanaSim](https://github.com/jiroshimaya/kanasim)によるランキング

```python
from soramimi_phonetic_search_dataset import rank_by_mora_editdistance

recall = evaluate_ranking_function(ranking_func=rank_by_mora_editdistance, topn=10)
```

## ライセンス

- **ソースコード**は**MITライセンス**の下で提供されています。詳細は[`LICENSE-CODE`](https://github.com/jiroshimaya/soramimi-phonetic-search-dataset/blob/main/LICENSE-CODE)をご覧ください。
- **データセット**は**CDLA-Permissive-2.0**の下で提供されています。詳細は[`src/soramimi_phonetic_search_dataset/data/LICENSE`](https://github.com/jiroshimaya/soramimi-phonetic-search-dataset/blob/main/src/soramimi_phonetic_search_dataset/data/LICENSE)をご覧ください。

## データセット使用上の注意

- このデータセットは、歌詞や実在の人名・キャラクター等の単語リストといった第三者のコンテンツを含んでいます。歌詞データは文節単位で分割・再構成され、元の文脈を復元できないよう慎重に加工されています。また、単語リストは一般に公開されている情報のみを使用しています。

- 研究目的での利用については、商用・非商用を問わず可能です。ただし、以下の点にご注意ください：
  - データセットの利用に関する法令遵守はユーザーの責任となります
  - データセットから元の歌詞を復元する行為は避けてください
  - 単語リストは音韻検索の評価以外の目的での使用はお控えください

## 引用

このデータセットを引用する場合は、以下の形式を使用してください：

```
@inproceedings{島谷2025soramimi,  
  author={島谷 二郎},  
  title={「〇〇で歌ってみた」替え歌を用いた音韻類似単語検索ベンチマークの構築},  
  booktitle={言語処理学会第31回年次大会 併設ワークショップ JLR2025},
  url={https://github.com/jiroshimaya/soramimi-phonetic-search-dataset},  
  year={2025},  
  month={3},  
}
```
