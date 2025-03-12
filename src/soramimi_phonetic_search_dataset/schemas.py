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


@dataclass
class PhoneticSearchMetrics:
    recall: float
    execution_time: float | None = None


@dataclass
class PhoneticSearchParameters:
    topn: int
    rank_func: str
    vowel_ratio: float | None = None
    rerank: bool = False
    rerank_model_name: str | None = None
    rerank_input_size: int | None = None


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
