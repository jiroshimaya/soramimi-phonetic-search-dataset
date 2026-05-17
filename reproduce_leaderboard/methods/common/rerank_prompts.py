from dataclasses import dataclass


DEFAULT_PROMPT_EXAMPLE_SUFFIX = """
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

DEFAULT_USER_PROMPT_TEMPLATE = """
Query: {query}
Wordlist:
{wordlist}
Top N: {topn}
Reranked:
"""


@dataclass(frozen=True)
class RerankPromptConfig:
    prompt_instructions: str
    prompt_example_suffix: str = DEFAULT_PROMPT_EXAMPLE_SUFFIX
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE
    requires_thoughts: bool = False


PROMPT_CONFIGS = {
    "default": RerankPromptConfig(
        prompt_instructions="""
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
    ),
    "simple": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
出力は上位Top N件のインデックスのみ返してください。
""",
    ),
    "detailed": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
- 子音より母音の一致を優先してください
- クエリとモウラ数が同じであることを優先してください。ただし促音（ッ）、撥音（ン）、長音（「ー」や直前のカナの母音と同じ単母音モウラ、エ段のカナの直後のイ、オ段のカナの直後のウ、など）の挿入や削除は許容されます。
出力は上位Top N件のインデックスのみ返してください。
""",
    ),
    "step_by_step": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
以下の手順で判断してください。
- 1. クエリと比較対象単語から促音（ッ）、撥音（ン）、長音（ー）を削除
- 2. クエリと比較対象単語をそれぞれ小文字ローマ字に直す
- 3. 同じ母音が連続していれば2文字目以降を削除する。例えば「k a a」は「k a」にする。「カア」は実質「カー」であるため長音の削除に相当。同様に「ei」「ou」についてはそれぞれ「e」「o」にする。これも「エイ」「オウ」は実質「エー」「オー」であるため長音の削除に対応する
- 4. 母音（aiueo）の並びが一致していることを優先し、母音の一致が同程度であればなるべく子音が似ているものを、より発音が似ているとする。
出力は上位Top N件のインデックスのみ返してください。
""",
    ),
    "detailed_romaji_explicit": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
- Query と Wordlist は、元のカタカナ表記をローマ字変換したものです
- 子音より母音の一致を優先してください
- クエリとモウラ数が同じであることを優先してください。ただし促音（ッ）、撥音（ン）、長音（「ー」や直前のカナの母音と同じ単母音モウラ、エ段のカナの直後のイ、オ段のカナの直後のウ、など）の挿入や削除は許容されます。
出力は上位Top N件のインデックスのみ返してください。
""",
    ),
    "nonreasoning_cot": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
以下の手順で判断してください。
- 1. クエリと比較対象単語から促音（ッ）、撥音（ン）、長音（ー）を削除
- 2. クエリと比較対象単語をそれぞれ小文字ローマ字に直す
- 3. 同じ母音が連続していれば2文字目以降を削除する。例えば「k a a」は「k a」にする。「カア」は実質「カー」であるため長音の削除に相当。同様に「ei」「ou」についてはそれぞれ「e」「o」にする。これも「エイ」「オウ」は実質「エー」「オー」であるため長音の削除に対応する
- 4. 母音（aiueo）の並びが一致していることを優先し、母音の一致が同程度であればなるべく子音が似ているものを、より発音が似ているとする。
構造化出力の thoughts フィールドには、最終順位に効いた判断要点だけを短い箇条書きで入れてください。
構造化出力の reranked フィールドには、上位Top N件のインデックスのみを入れてください。
""",
        requires_thoughts=True,
    ),
}


def get_prompt_config(prompt_template: str = "default") -> RerankPromptConfig:
    try:
        return PROMPT_CONFIGS[prompt_template]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt_template: {prompt_template}") from exc