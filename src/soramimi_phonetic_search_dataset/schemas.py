from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhoneticSearchQuery:
    query: str
    positive: list[str]


@dataclass
class PhoneticSearchDataset:
    queries: list[PhoneticSearchQuery]
    words: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhoneticSearchDataset":
        queries = [PhoneticSearchQuery(**query) for query in data["queries"]]
        words = data["words"]
        metadata = data.get("metadata", {})
        return cls(queries=queries, words=words, metadata=metadata)


@dataclass
class PhoneticSearchResult:
    query: str
    ranked_words: list[str]
    positive_words: list[str]
    thoughts: list[str] | None = None


@dataclass
class PhoneticSearchMetrics:
    recall: float
    execution_time: float
    rerank_input_tokens: int | None = None
    rerank_output_tokens: int | None = None
    rerank_reasoning_tokens: int | None = None
    rerank_total_tokens: int | None = None
    rerank_input_cost: float | None = None
    rerank_output_cost: float | None = None
    rerank_reasoning_cost: float | None = None
    rerank_total_cost: float | None = None


@dataclass
class PhoneticSearchParameters:
    topn: int
    rank_func: str
    query_limit: int | None = None
    query_offset: int | None = None
    vowel_ratio: float | None = None
    rerank: bool = False
    rerank_backend: str | None = None
    rerank_model_name: str | None = None
    rerank_batch_id: str | None = None
    rerank_reasoning_effort: str | None = None
    rerank_prompt_template: str | None = None
    rerank_include_thoughts: bool | None = None
    rerank_input_transform: str | None = None
    rerank_input_size: int | None = None
    execution_timestamp: str | None = None


@dataclass
class PhoneticSearchResults:
    parameters: PhoneticSearchParameters
    metrics: PhoneticSearchMetrics
    results: list[PhoneticSearchResult]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhoneticSearchResults":
        results = [PhoneticSearchResult(**result) for result in data["results"]]
        return cls(
            parameters=PhoneticSearchParameters(**data["parameters"]),
            metrics=PhoneticSearchMetrics(**data["metrics"]),
            results=results,
        )
