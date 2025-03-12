# Reproduce Leaderboard

soramimi-phonetic-search-datasetのleaderboardの再現用スクリプト群です。

## 概要

[leaderboard.md](../leaderboard.md)に記載されている各手法のRecall@10を再現するためのスクリプトです。

## 使い方

### 全ての手法を実行

```bash
sh run_all.sh
```

### 個別の手法を実行

```bash
uv run methods/000_mora.py  # モーラ編集距離
uv run methods/001_phoneme.py  # 音素編集距離
uv run methods/002_vowel_consonant.py  # 母音子音編集距離
uv run methods/003_kanasim.py  # KanaSim編集距離
uv run methods/004_llm_rerank_gpt4o_mini.py  # LLMリランク (gpt-4o-mini)
uv run methods/005_llm_rerank_gpt4o.py  # LLMリランク (gpt-4o)
uv run methods/006_llm_rerank_gemini.py  # LLMリランク (gemini-2.0-flash)
uv run methods/007_llm_rerank_gpt45preview.py  # LLMリランク (gpt-4.5-preview)
```


## 結果の出力

各手法の実行結果は`results/`ディレクトリに保存されます：

```
results/
├── 000_mora.json
├── 001_phoneme.json
├── 002_vowel_consonant.json
├── 003_kanasim.json
├── 004_llm_rerank_gpt4o_mini.json
├── 005_llm_rerank_gpt4o.json
├── 006_llm_rerank_gemini.json
└── 007_llm_rerank_gpt45preview.json
``` 