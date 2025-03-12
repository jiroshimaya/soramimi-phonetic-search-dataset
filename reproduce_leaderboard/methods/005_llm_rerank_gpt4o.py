"""
LLMリランク (gpt-4o) による評価を実行するスクリプト
"""

import subprocess
from pathlib import Path


def main():
    # 結果の出力先を作成
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "005_llm_rerank_gpt4o.json"

    # evaluate_ranking.pyを実行
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
        "--rerank_model_name",
        "gpt-4o",
        "--rerank_input_size",
        "100",
        "--rerank_batch_size",  # tier1の制限内で動作させるため
        "2",
        "--rerank_interval",  # tier1の制限内で動作させるため
        "1",
        "--output_file_path",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
