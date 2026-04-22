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
| LLM Rerank (gemini-2.0-flash) | - | - | - | API key 未設定のため未計測 |
| LLM Rerank (gpt-4.5-preview) | - | - | - | 現行 LiteLLM では provider 解決に失敗し未計測 |
| LLM Rerank (gpt-5.4) | 0.600 | 3.8 | 0.0272 | default prompt |
| LLM Rerank (gpt-5.4, prompt 008_01 simple) | 0.550 | 5.8 | 0.0254 | `008_01_simple` |
| LLM Rerank (gpt-5.4, prompt 008_02 detailed) | 0.450 | 3.7 | 0.0281 | `008_02_detailed` |
| LLM Rerank (gpt-5.4, prompt 008_03 step-by-step) | 0.650 | 3.5 | 0.0312 | `008_03_step_by_step` |
| LLM Rerank (gpt-5.4, medium, prompt 010_01 simple) | 0.650 | 126.0 | 0.6511 | `reasoning_effort=medium` |
| LLM Rerank (gpt-5.4, medium, prompt 010_02 detailed) | 0.900 | 230.0 | 0.9312 | `reasoning_effort=medium` |
| LLM Rerank (gpt-5.4, medium, prompt 010_03 step-by-step) | 1.000 | 186.4 | 1.0800 | `reasoning_effort=medium` |

## メモ

- 実測日時: 2026-04-22
- API Cost は各 small run の `rerank_total_cost` を丸めた値です
- Time は `execution_time` の実測値で、ネットワークやプロバイダ状態でぶれます
- full leaderboard の `*_cost_estimate.json` は **small での実測から 150 件へ外挿**したものです
