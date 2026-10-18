# Reproduce Leaderboard

`leaderboard.md` に掲載している代表的な評価結果を再現するためのスクリプト群です。

細かな prompt/input の派生実験や probe スクリプトは、本家から切り離して [soramimi-phonetic-search-experiments](https://github.com/jiroshimaya/soramimi-phonetic-search-experiments) で管理します。

## インストール

プロジェクトのルートで以下を実行します。

```bash
uv pip install -e .
uv pip install --group eval
```

## 実行方法

### すべての代表手法を実行

```bash
cd reproduce_leaderboard
sh run_all.sh
```

### 個別実行

```bash
cd reproduce_leaderboard

uv run methods/000_mora.py
uv run methods/001_phoneme.py
uv run methods/002_vowel_consonant.py
uv run methods/003_kanasim.py
uv run methods/004_llm_rerank_gpt4o_mini.py
uv run methods/005_llm_rerank_gpt4o.py
uv run methods/006_llm_rerank_gemini.py
uv run methods/007_llm_rerank_gpt45preview.py
uv run methods/008_llm_rerank_gpt54.py
uv run methods/010_llm_rerank_gpt54_medium_step_by_step.py
```

### カスタム評価

細かな派生実験やアドホックな評価は、このリポジトリではなく
[soramimi-phonetic-search-experiments](https://github.com/jiroshimaya/soramimi-phonetic-search-experiments)
またはパッケージ API を使って行ってください。

## 出力

- `results/`: full dataset の結果
- `results_small/`: small dataset の結果
- `leaderboard.small.md`: 先頭10クエリだけでの軽量版 leaderboard

現在このリポジトリに残している代表的な結果ファイルは次のとおりです。

```text
results/
├── 000_mora.json
├── 001_phoneme.json
├── 002_vowel_consonant.json
├── 003_kanasim.json
├── 004_llm_rerank_gpt4o_mini.json
├── 005_llm_rerank_gpt4o.json
├── 006_llm_rerank_gemini.json
├── 007_llm_rerank_gpt45preview.json
├── 008_llm_rerank_gpt54.json
├── 010_llm_rerank_gpt54_medium_step_by_step.json
└── 010_llm_rerank_gpt54_medium_step_by_step_cost_estimate.json
```

## subset ラベル更新

`baseball.json` の `subset` と、保存済み結果 JSON 内の `recall_by_subset` は
次のスクリプトでまとめて再生成できます。

```bash
uv run python scripts/update_subset_labels.py
```

## 注意事項

- 評価には `baseball.json` データセットが使われます
- `--dataset_size small` を使うと、同じ単語リストのまま先頭10クエリだけで評価できます
- LLM を使う場合は `OPENAI_API_KEY` / `GEMINI_API_KEY` などの環境変数を設定してください
