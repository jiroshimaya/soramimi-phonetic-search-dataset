import time
from typing import Any, Type

from litellm import batch_completion, completion
from pydantic import BaseModel
from tqdm import tqdm

PROMPT_INSTRUCTIONS = {
    "default": """
    You are a phonetic search assistant.
    You are given a query and a list of words.
    You need to rerank the words based on phonetic similarity to the query.
    When estimating phonetic similarity, please consider the following:
    1. Prioritize matching vowels
    2. Substitution, insertion, or deletion of nasal sounds, geminate consonants, and long vowels is acceptable
    3. For other cases, words with similar mora counts are preferred
    You need to return only the reranked list of index numbers of the words, no other text.
    You need to return only topn index numbers.
    """,
    "008_01_simple": """
    クエリ（Query）と単語一覧（Wordlist）が与えられます。
    クエリと発音が似ている順に、単語一覧を並び替えてください。
    出力は上位Top N件のインデックスのみ返してください。
    """,
    "008_02_detailed": """
    クエリ（Query）と単語一覧（Wordlist）が与えられます。
    クエリと発音が似ている順に、単語一覧を並び替えてください。
    - 子音より母音の一致を優先してください
    - クエリとモウラ数が同じであることを優先してください。ただし促音（ッ）、撥音（ン）、長音（「ー」や直前のカナの母音と同じ単母音モウラ、エ段のカナの直後のイ、オ段のカナの直後のウ、など）の挿入や削除は許容されます。
    出力は上位Top N件のインデックスのみ返してください。
    """,
    "008_03_step_by_step": """
    クエリ（Query）と単語一覧（Wordlist）が与えられます。
    クエリと発音が似ている順に、単語一覧を並び替えてください。
    以下の手順で判断してください。
    - 1. クエリと比較対象単語から促音（ッ）、撥音（ン）、長音（ー）を削除
    - 2. クエリと比較対象単語をそれぞれ小文字ローマ字に直す
    - 3. 同じ母音が連続していれば2文字目以降を削除する。例えば「k a a」は「k a」にする。「カア」は実質「カー」であるため長音の削除に相当。同様に「ei」「ou」についてはそれぞれ「e」「o」にする。これも「エイ」「オウ」は実質「エー」「オー」であるため長音の削除に対応する
    - 4. 母音（aiueo）の並びが一致していることを優先し、母音の一致が同程度であればなるべく子音が似ているものを、より発音が似ているとする。
    出力は上位Top N件のインデックスのみ返してください。
    """,
}

PROMPT_EXAMPLE_SUFFIX = """
Example:
Query: タロウ
Wordlist:
0. アオ
1. アオウヅ
2. アノウ
3. タキョウ
4. タド
5. タノ
6. タロウ
7. タンノ
Top N: 5
Reranked: 6, 4, 5, 7, 2
"""


def build_system_prompt(prompt_template: str = "default") -> str:
    try:
        prompt_instructions = PROMPT_INSTRUCTIONS[prompt_template]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt_template: {prompt_template}") from exc
    return f"{prompt_instructions.strip()}\n\n{PROMPT_EXAMPLE_SUFFIX.strip()}"


def get_structured_outputs(
    model_name: str,
    messages: list[list[dict[str, Any]]],
    response_format: Type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    reasoning_effort: str | None = None,
) -> list[BaseModel]:
    is_gpt5 = model_name.startswith("gpt-5")
    normalized_reasoning_effort = (
        None if reasoning_effort in (None, "none") else reasoning_effort
    )

    completion_kwargs: dict[str, Any] = {}
    if is_gpt5:
        completion_kwargs["max_completion_tokens"] = max_tokens
        if normalized_reasoning_effort is not None:
            completion_kwargs["extra_body"] = {
                "reasoning_effort": normalized_reasoning_effort
            }
    else:
        completion_kwargs["temperature"] = temperature
        completion_kwargs["max_tokens"] = max_tokens
        if normalized_reasoning_effort is not None:
            completion_kwargs["reasoning_effort"] = normalized_reasoning_effort

    def parse_response(response: Any) -> BaseModel:
        if not hasattr(response, "choices"):
            raise TypeError(f"Unexpected LiteLLM response: {response!r}")

        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"Empty LiteLLM content: {response!r}")
        return response_format.model_validate_json(content)

    raw_responses = batch_completion(
        model=model_name,
        messages=messages,
        response_format=response_format,
        **completion_kwargs,
    )

    parsed_responses = []
    for message, response in zip(messages, raw_responses):
        try:
            parsed_responses.append(parse_response(response))
        except (TypeError, ValueError):
            fallback_kwargs = completion_kwargs.copy()
            if is_gpt5:
                fallback_kwargs["max_completion_tokens"] = max(max_tokens, 4000)
            fallback_response = completion(
                model=model_name,
                messages=message,
                response_format=response_format,
                **fallback_kwargs,
            )
            parsed_responses.append(parse_response(fallback_response))
    return parsed_responses


def rerank_by_llm(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int = 10,
    model_name: str = "gpt-4o-mini",
    reasoning_effort: str | None = None,
    prompt_template: str = "default",
    batch_size: int = 10,
    temperature: float = 0.0,
    rerank_interval: int = 60,
) -> list[list[str]]:
    class RerankedWordlist(BaseModel):
        reranked: list[int]

    prompt = build_system_prompt(prompt_template)

    user_prompt = """
    Query: {query}
    Wordlist:
    {wordlist}
    Top N: {topn}
    Reranked:
    """

    messages = []
    for query, wordlist in zip(query_texts, wordlist_texts):
        wordlist_str = "\n".join([f"{i}. {word}" for i, word in enumerate(wordlist)])
        message = []
        message.append({"role": "system", "content": prompt})
        message.append(
            {
                "role": "user",
                "content": user_prompt.format(
                    query=query, wordlist=wordlist_str, topn=topn
                ),
            }
        )
        messages.append(message)

    reranked_wordlists = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch_messages = messages[i : i + batch_size]
        responses = get_structured_outputs(
            model_name=model_name,
            messages=batch_messages,
            temperature=temperature,
            max_tokens=1000,
            response_format=RerankedWordlist,
            reasoning_effort=reasoning_effort,
        )
        for wordlist, response in zip(wordlist_texts[i : i + batch_size], responses):
            reranked_wordlist = []

            response_typed = RerankedWordlist.model_validate(response)
            for i in response_typed.reranked:
                if 0 <= i < len(wordlist):
                    reranked_wordlist.append(wordlist[i])
                else:
                    reranked_wordlist.append("NA")
            reranked_wordlists.append(reranked_wordlist)

        time.sleep(rerank_interval)

    return reranked_wordlists
