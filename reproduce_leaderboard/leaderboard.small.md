# Small Leaderboard

開発者・研究者が手元で試行錯誤しやすいように、**先頭10クエリ**だけで評価した軽量版 leaderboard です。

- 正式な比較や引用には、リポジトリ直下の [`leaderboard.md`](../leaderboard.md) を使ってください
- こちらは **small dataset (`--dataset_size small`)** を使ったスモークテスト寄りの結果です
- 目立ちすぎないよう、`reproduce_leaderboard/` 配下にのみ置いています

| Method | Recall@10 | Time (s) | API Cost (USD) | Notes |
|--------|-----------|----------|----------------|-------|
| Mora EditDistance | 0.450 | 0.3 | - | 先頭10クエリ |
| Phoneme EditDistance | 0.750 | 0.6 | - | 先頭10クエリ |
| Vowel Consonant EditDistance | 0.750 | 0.5 | - | `vowel_ratio=0.8` |
| KanaSim EditDistance | 0.650 | 1.9 | - | `vowel_ratio=0.8` |
| LLM Rerank (gpt-4o-mini) | 0.500 | 3.9 | 0.0015 | `gpt-4o-mini` |
| LLM Rerank (gpt-4o) | 0.650 | 2.4 | 0.0246 | `gpt-4o` |
| LLM Rerank (gemini-2.0-flash) | 0.550 | - | - | score は `results/006_llm_rerank_gemini.json` の先頭10件から補完 |
| LLM Rerank (gpt-4.5-preview) | 0.750 | - | - | score は `results/007_llm_rerank_gpt45preview.json` の先頭10件から補完 |
| LLM Rerank (gpt-5.4) | 0.600 | 3.8 | 0.0272 | default prompt |
| LLM Rerank (gpt-5.4, prompt simple) | 0.550 | 5.8 | 0.0254 | `simple` |
| LLM Rerank (gpt-5.4, prompt detailed) | 0.450 | 3.7 | 0.0281 | `detailed` |
| LLM Rerank (gpt-5.4, prompt step-by-step) | 0.650 | 3.5 | 0.0312 | `step_by_step` |
| LLM Rerank (gpt-5.4, prompt detailed, pyopenjtalk romaji input) | 0.650 | 4.0 | 0.0328 | `detailed` + `pyopenjtalk_romaji` |
| LLM Rerank (gpt-5.4, prompt detailed romaji explicit, pyopenjtalk romaji input) | 0.650 | 3.7 | 0.0334 | `detailed_romaji_explicit` + `pyopenjtalk_romaji` |
| LLM Rerank (gpt-5.4, prompt detailed, kana+pyopenjtalk romaji input) | 0.750 | 3.5 | 0.0424 | `detailed` + `kana_and_pyopenjtalk_romaji` |
| LLM Rerank (gpt-5.4, non-reasoning CoT) | 0.800 | 6.9 | 0.0585 | `nonreasoning_cot` + `thoughts` |
| LLM Rerank (gpt-5.4, medium, prompt 010_01 simple) | 0.650 | 126.0 | 0.6511 | `reasoning_effort=medium` |
| LLM Rerank (gpt-5.4, medium, prompt 010_02 detailed) | 0.900 | 230.0 | 0.9312 | `reasoning_effort=medium` |
| LLM Rerank (gpt-5.4, medium, prompt 010_03 step-by-step) | 1.000 | 186.4 | 1.0800 | `reasoning_effort=medium` |

## 参考: 同条件を先頭100クエリで実測

small leaderboard 本体は先頭10クエリですが、同条件をもう少し安定して見たいとき向けに **先頭100クエリ** の実測も残しておきます。

| Method | Recall@10 | Time (s) | API Cost (USD) | Notes |
|--------|-----------|----------|----------------|-------|
| LLM Rerank (gpt-5.4, prompt detailed, kana+pyopenjtalk romaji input) | 0.550 | 24.4 | 0.0454 | `detailed` + `kana_and_pyopenjtalk_romaji`; `query_limit=100` |

## メモ

- small table の実測日時: 2026-04-22
- API Cost は各 small run の `rerank_total_cost` を丸めた値です
- Time は `execution_time` の実測値で、ネットワークやプロバイダ状態でぶれます
- `LLM Rerank (gpt-5.4, non-reasoning CoT)` は 2026-04-26 実測で、`results_small/008_07_llm_rerank_gpt54_nonreasoning_cot.json` に対応します
- `gemini-2.0-flash` と `gpt-4.5-preview` は small 実測がないため、Recall@10 のみ `results/*.json` の先頭10件から補完しています
- full leaderboard の `*_cost_estimate.json` は **small での実測から 150 件へ外挿**したものです
- 先頭100クエリ実測は 2026-04-26 実行で、`results_first100/008_06_llm_rerank_gpt54_detailed_kana_and_pyopenjtalk_romaji.json` に対応します
