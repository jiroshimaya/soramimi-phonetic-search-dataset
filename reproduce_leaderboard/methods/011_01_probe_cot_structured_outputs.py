"""
Responses API の structured outputs を使って、visible な thoughts と reranked、
および gpt-5 系では reasoning.summary を観測するための単発検証スクリプト。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reproduce_leaderboard.methods.common.reranker import PROMPT_INSTRUCTIONS
from soramimi_phonetic_search_dataset import load_small_dataset
from soramimi_phonetic_search_dataset import rank_by_vowel_consonant_editdistance


DEFAULT_MODELS = ["gpt-5.4", "gpt-4.1"]


class StructuredRerankOutput(BaseModel):
    thoughts: list[str] = Field(
        description=(
            "Visible intermediate notes. Keep them short and concrete, and summarize "
            "what was compared without exposing private hidden reasoning."
        )
    )
    reranked: list[int] = Field(
        description="Top N candidate indices sorted by phonetic similarity."
    )


@dataclass
class ProbeInput:
    query: str
    positive_words: list[str]
    wordlist: list[str]


def build_probe_input(query_index: int, candidate_size: int, vowel_ratio: float) -> ProbeInput:
    dataset = load_small_dataset()
    if not 0 <= query_index < len(dataset.queries):
        raise ValueError(
            f"query_index must be between 0 and {len(dataset.queries) - 1}, got {query_index}"
        )

    target_query = dataset.queries[query_index]
    ranked_wordlist = rank_by_vowel_consonant_editdistance(
        [target_query.query], dataset.words, vowel_ratio=vowel_ratio
    )[0]
    candidate_wordlist = ranked_wordlist[:candidate_size]

    missing_positive_words = [
        word for word in target_query.positive if word not in candidate_wordlist
    ]
    if missing_positive_words:
        candidate_wordlist = candidate_wordlist[: max(candidate_size - len(missing_positive_words), 0)]
        for word in missing_positive_words:
            if word not in candidate_wordlist:
                candidate_wordlist.append(word)

    candidate_wordlist = sorted(candidate_wordlist)
    return ProbeInput(
        query=target_query.query,
        positive_words=target_query.positive,
        wordlist=candidate_wordlist,
    )


def build_system_prompt() -> str:
    base_instruction = PROMPT_INSTRUCTIONS["008_03_step_by_step"].strip()
    return f"""
{base_instruction}

ただし最終出力は JSON object とし、以下の 2 項目を必ず含めてください。
- thoughts: 可視な中間メモの配列。各要素は短く具体的にし、比較に使った要点だけを書くこと
- reranked: 上位 Top N 件のインデックス配列

thoughts は hidden reasoning の逐語的な露出ではなく、人間が読める要約メモとして 3-6 項目程度で書いてください。
reranked には Wordlist のインデックスのみを含めてください。

Example JSON:
{{
  "thoughts": [
    "促音・撥音・長音を無視する前提で比較する",
    "クエリと候補をローマ字化して母音列を優先して見る",
    "母音列が近い候補同士では子音とモーラ数の近さで並べる"
  ],
  "reranked": [6, 4, 5, 7, 2]
}}
""".strip()


def build_user_prompt(probe_input: ProbeInput, topn: int) -> str:
    wordlist_lines = "\n".join(
        f"{index}. {word}" for index, word in enumerate(probe_input.wordlist)
    )
    return f"""
Query: {probe_input.query}
Wordlist:
{wordlist_lines}
Top N: {topn}
""".strip()


def extract_reasoning_summaries(response: Any) -> list[str]:
    summaries: list[str] = []
    for output in response.output:
        if output.type != "reasoning":
            continue
        summaries.extend(summary.text for summary in output.summary)
    return summaries


def usage_to_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    output_tokens_details = getattr(usage, "output_tokens_details", None)
    input_tokens_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(output_tokens_details, "reasoning_tokens", None),
        "cached_input_tokens": getattr(input_tokens_details, "cached_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def reranked_words(wordlist: list[str], reranked_indices: list[int]) -> list[str]:
    return [wordlist[index] for index in reranked_indices]


def run_probe(
    *,
    client: OpenAI,
    model_name: str,
    probe_input: ProbeInput,
    topn: int,
    max_output_tokens: int,
    reasoning_effort: str,
    reasoning_summary: str,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": build_system_prompt(),
        "input": build_user_prompt(probe_input, topn),
        "text_format": StructuredRerankOutput,
        "max_output_tokens": max_output_tokens,
    }
    if model_name.startswith("gpt-5"):
        request_kwargs["reasoning"] = {
            "effort": reasoning_effort,
            "summary": reasoning_summary,
        }

    response = client.responses.parse(**request_kwargs)
    parsed_output = response.output_parsed
    if parsed_output is None:
        raise ValueError(f"Structured output was empty for {model_name}")

    reasoning_summaries = extract_reasoning_summaries(response)
    reranked_wordlist = reranked_words(probe_input.wordlist, parsed_output.reranked)
    positive_hit_count = sum(
        1 for word in probe_input.positive_words if word in reranked_wordlist[:topn]
    )

    return {
        "model": model_name,
        "reasoning": request_kwargs.get("reasoning"),
        "response_id": response.id,
        "status": response.status,
        "incomplete_details": (
            response.incomplete_details.to_dict()
            if response.incomplete_details is not None
            else None
        ),
        "structured_output": parsed_output.model_dump(),
        "reasoning_summaries": reasoning_summaries,
        "usage": usage_to_dict(response),
        "reranked_words": reranked_wordlist,
        "positive_words": probe_input.positive_words,
        "positive_hit_count_in_topn": positive_hit_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Responses API で structured outputs と reasoning.summary を観測する "
            "単発検証スクリプト"
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="比較するモデル一覧。デフォルト: gpt-5.4 gpt-4.1",
    )
    parser.add_argument(
        "--query_index",
        type=int,
        default=1,
        help="small dataset 内の query index。デフォルト: 1 (アウマル)",
    )
    parser.add_argument(
        "--candidate_size",
        type=int,
        default=20,
        help="母音子音ランキングから拾う候補数",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="構造化出力に要求する上位件数",
    )
    parser.add_argument(
        "--vowel_ratio",
        type=float,
        default=0.5,
        help="候補生成に使う母音重み",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="none",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="gpt-5 系に渡す reasoning.effort",
    )
    parser.add_argument(
        "--reasoning_summary",
        type=str,
        default="detailed",
        choices=["auto", "concise", "detailed"],
        help="gpt-5 系に渡す reasoning.summary",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=2000,
        help="Responses API の max_output_tokens",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "011_01_probe_cot_structured_outputs.json",
        help="結果 JSON の保存先",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_file_path.parent.mkdir(parents=True, exist_ok=True)

    probe_input = build_probe_input(
        query_index=args.query_index,
        candidate_size=args.candidate_size,
        vowel_ratio=args.vowel_ratio,
    )
    client = OpenAI()

    results = []
    for model_name in args.models:
        results.append(
            run_probe(
                client=client,
                model_name=model_name,
                probe_input=probe_input,
                topn=args.topn,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                reasoning_summary=args.reasoning_summary,
            )
        )

    output = {
        "probe_input": {
            **asdict(probe_input),
            "topn": args.topn,
            "query_index": args.query_index,
            "candidate_size": args.candidate_size,
            "vowel_ratio": args.vowel_ratio,
        },
        "results": results,
    }

    with args.output_file_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
