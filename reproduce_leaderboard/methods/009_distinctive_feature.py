"""
pyphone による弁別素性ベース距離の評価を実行するスクリプト
"""

import subprocess
from pathlib import Path


def main():
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "009_distinctive_feature.json"

    evaluate_script = Path(__file__).parent / "common" / "evaluate_ranking.py"
    cmd = [
        "uv",
        "run",
        str(evaluate_script),
        "--rank_func",
        "distinctive_feature",
        "--topn",
        "10",
        "--output_file_path",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
