"""
Gemini-2.0-flashによるリランクを実行するスクリプト
"""

import subprocess
from pathlib import Path


def main():
    # 結果の出力先を作成
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "006_llm_rerank_gemini.json"

    # basic_usage.pyを実行
    cmd = [
        "uv",
        "run",
        "examples/basic_usage.py",
        "--rank_func",
        "vowel_consonant",
        "--topn",
        "10",
        "--vowel_ratio",
        "0.5",
        "--rerank",
        "--rerank_model_name",
        "gemini/gemini-2.0-flash",
        "--rerank_input_size",
        "100",
        "--rerank_batch_size",  # 無料枠の制限内で動作させるため
        "15",
        "--rerank_interval",  # 無料枠の制限内で動作させるため
        "60",
        "--output_file_path",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
