#!/bin/bash

# 全ての手法を順番に実行

echo "=== Mora EditDistance ==="
uv run methods/000_mora.py

echo "=== Phoneme EditDistance ==="
uv run methods/001_phoneme.py

echo "=== Vowel Consonant EditDistance ==="
uv run methods/002_vowel_consonant.py

echo "=== KanaSim EditDistance ==="
uv run methods/003_kanasim.py

echo "=== LLM Rerank (gpt-4o-mini) ==="
uv run methods/004_llm_rerank_gpt4o_mini.py

echo "=== LLM Rerank (gpt-4o) ==="
uv run methods/005_llm_rerank_gpt4o.py

echo "=== LLM Rerank (gemini-2.0-flash) ==="
uv run methods/006_llm_rerank_gemini.py

echo "=== LLM Rerank (gpt-4.5-preview) ==="
uv run methods/007_llm_rerank_gpt45preview.py

echo "Done!" 