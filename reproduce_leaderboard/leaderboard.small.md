# Small Leaderboard

開発者・研究者が手元で試行錯誤しやすいように、**先頭10クエリ**だけで見た軽量版 leaderboard です。

- 正式な比較や引用には、リポジトリ直下の [`leaderboard.md`](../leaderboard.md) を使ってください
- こちらは既存の `reproduce_leaderboard/results/*.json` に入っている各 run の**先頭10件**から Recall@10 を再計算した補助ビューです
- 目立ちすぎないよう、`reproduce_leaderboard/` 配下にのみ置いています

| Method | Recall@10 | Notes |
|--------|-----------|-------|
| Mora EditDistance | 0.450 | `results/000_mora.json` の先頭10件 |
| Phoneme EditDistance | 0.750 | `results/001_phoneme.json` の先頭10件 |
| Vowel Consonant EditDistance | 0.750 | `results/002_vowel_consonant.json` の先頭10件 |
| KanaSim EditDistance | 0.650 | `results/003_kanasim.json` の先頭10件 |
| LLM Rerank (gpt-4o-mini) | 0.500 | `results/004_llm_rerank_gpt4o_mini.json` の先頭10件 |
| LLM Rerank (gpt-4o) | 0.750 | `results/005_llm_rerank_gpt4o.json` の先頭10件 |
| LLM Rerank (gemini-2.0-flash) | 0.550 | `results/006_llm_rerank_gemini.json` の先頭10件 |
| LLM Rerank (gpt-4.5-preview) | 0.750 | `results/007_llm_rerank_gpt45preview.json` の先頭10件 |
| LLM Rerank (gpt-5.4) | 0.650 | `results/008_llm_rerank_gpt54.json` の先頭10件 |
| LLM Rerank (gpt-5.4, prompt 008_01 simple) | 0.500 | `results/008_01_llm_rerank_gpt54_simple.json` の先頭10件 |
| LLM Rerank (gpt-5.4, prompt 008_02 detailed) | 0.600 | `results/008_02_llm_rerank_gpt54_detailed.json` の先頭10件 |
| LLM Rerank (gpt-5.4, prompt 008_03 step-by-step) | 0.600 | `results/008_03_llm_rerank_gpt54_step_by_step.json` の先頭10件 |
| LLM Rerank (gpt-5.4, medium, prompt 010_01 simple) | 0.750 | `results/010_01_llm_rerank_gpt54_medium_simple.json` の先頭10件 |
| LLM Rerank (gpt-5.4, medium, prompt 010_02 detailed) | 1.000 | `results/010_02_llm_rerank_gpt54_medium_detailed.json` の先頭10件 |
| LLM Rerank (gpt-5.4, medium, prompt 010_03 step-by-step) | 1.000 | `results/010_03_llm_rerank_gpt54_medium_step_by_step.json` の先頭10件 |

## メモ

- これは **small dataset を新規実行した結果ではなく**、既存の full run 結果を query 単位で先頭10件に切って再集計した値です
- そのため、Time や API Cost はこの表には載せていません
- LLM 系は run ごとにゆらぎうるため、`--dataset_size small` で再実行した値と完全一致しない場合があります
