"""
GPT-4.5-previewによるLLMリランクを実行するスクリプト
"""

import subprocess
from pathlib import Path


def main():
    # 結果の出力先を作成
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "007_llm_rerank_gpt45preview.json"

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
        "--rerank_input_size",
        "100",
        "--rerank_model_name",
        "gpt-4.5-preview",
        "--output_file_path",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
