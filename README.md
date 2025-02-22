# soramimi-phonetic-search-dataset
A dataset based on soramimi parody songs to evaluate phonetic search system

# Usage

```
uv run src/evaluate_phonetic_search_dataset.py -i data/baseball.json
```
# License
- The **source code** is licensed under the **MIT License**. See [`src/LICENSE`](src/LICENSE).
- The **dataset** is licensed under the **CDLA-Permissive-2.0**. See [`data/LICENSE`](data/LICENSE).

# Dataset Usage Notes

-	This dataset may contain fragments of lyrics or word lists that originate from third-party content and may be subject to copyright or other intellectual property rights.
-	It is intended for research and development of phonetic search technologies. Its use as a phonetic search dataset is considered non-problematic; however, users are responsible for ensuring compliance with applicable copyright, trademark, and publicity rights laws.
- The dataset must not be used to reconstruct original lyrics or create derivative works that may infringe intellectual property rights.

# Citation
If you wish to cite this dataset:

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

If you need to cite in English, please use the following:

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