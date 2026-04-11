"""
LLMリランク (gpt-5.4, reasoning effort medium, step-by-step prompt) による評価を実行するスクリプト
"""

import subprocess
from pathlib import Path


def main():
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "010_03_llm_rerank_gpt54_medium_step_by_step.json"

    evaluate_script = Path(__file__).parent / "common" / "evaluate_ranking.py"
    cmd = [
        "uv",
        "run",
        str(evaluate_script),
        "--rank_func",
        "vowel_consonant",
        "--topn",
        "10",
        "--vowel_ratio",
        "0.5",
        "--rerank",
        "--rerank_input_size",
        "100",
        "--rerank_interval",
        "0",
        "--rerank_batch_size",
        "10",
        "--rerank_model_name",
        "gpt-5.4",
        "--rerank_reasoning_effort",
        "medium",
        "--rerank_prompt_template",
        "008_03_step_by_step",
        "--output_file_path",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
