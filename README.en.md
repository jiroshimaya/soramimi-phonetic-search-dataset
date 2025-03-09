# soramimi-phonetic-search-dataset

A dataset for evaluating phonetic search systems, built from Japanese homophonic translation parody songs ("soramimi") that recreate lyrics using only words from specific genres.

## Overview

This dataset is designed for evaluating phonetic search systems, particularly for Japanese language. It contains word pairs extracted from homophonic translation parody songs (where lyrics are recreated using only words from a specific genre while maintaining similar sounds), making it particularly suitable for evaluating phonetic search systems that emphasize matching rhymes and rhythms.

For evaluation results of different methods, please see the [leaderboard](https://github.com/jiroshimaya/soramimi-phonetic-search-dataset/blob/main/leaderboard.md).

## Usage

### Installation

```bash
pip install soramimi-phonetic-search-dataset
```

### Basic Usage

```python
from soramimi_phonetic_search_dataset import evaluate_ranking_function

# Define your custom ranking function
def my_ranking_function(query_texts: list[str], wordlist_texts: list[str]) -> list[list[str]]:
    # Implement your phonetic similarity ranking logic here
    return ranked_wordlists

# Evaluate your function
recall = evaluate_ranking_function(ranking_func=my_ranking_function, topn=10)
print(f"Recall@10: {recall}")
```

### Sample Ranking Functions

The following ranking functions are provided:

- `rank_by_mora_editdistance`: Ranking based on mora edit distance
- `rank_by_vowel_consonant_editdistance`: Ranking based on separate vowel and consonant edit distances
- `rank_by_phoneme_editdistance`: Ranking based on phoneme edit distance
- `rank_by_kanasim`: Ranking using KanaSim algorithm

```python
from soramimi_phonetic_search_dataset import rank_by_mora_editdistance

recall = evaluate_ranking_function(ranking_func=rank_by_mora_editdistance, topn=10)
```

## Dataset Usage Notes

- This dataset contains third-party content including lyrics and word lists of real names and characters. The lyric data has been carefully processed by splitting and restructuring at the phrase level to prevent reconstruction of the original context. The word lists only use publicly available information.

- The dataset can be used for research purposes, both commercial and non-commercial. However, please note the following:
  - Users are responsible for compliance with applicable laws regarding dataset usage
  - Please refrain from attempting to reconstruct original lyrics from the dataset
  - Word lists should only be used for evaluating phonetic search systems

## License

- The **source code** is licensed under the **MIT License**. See [`LICENSE-CODE`](LICENSE-CODE).
- The **dataset** is licensed under the **CDLA-Permissive-2.0**. See [`src/soramimi_phonetic_search_dataset/data/LICENSE`](src/soramimi_phonetic_search_dataset/data/LICENSE).

## Citation

```
@inproceedings{shimaya2025soramimi,  
  author={Jiro Shimaya},  
  title={Phonetic word search benchmark based on homophonic parody song using only words from a specific genre.},  
  booktitle={NLP2025 Workshop on Japanese Language Resources (JLR2025)},
  url={https://github.com/jiroshimaya/soramimi-phonetic-search-dataset},  
  year={2025},  
  month={3},  
}
```

日本語版のREADMEは[こちら](README.ja.md)をご覧ください。
