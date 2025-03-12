"""
母音子音編集距離による評価を実行するスクリプト
"""

import subprocess
from pathlib import Path


def main():
    # 結果の出力先を作成
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "002_vowel_consonant.json"

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
        "0.8",  # leaderboardの設定値
        "--output_file_path",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
