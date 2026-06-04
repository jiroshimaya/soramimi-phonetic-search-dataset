# ADR 00003: Store query difficulty labels

- Status: accepted
- Date: 2026-06-04
- Supersedes: none
- Superseded by: none

## Context

`baseball.json` に難易度サブセットを導入したい。難易度は `easy` / `medium` / `hard` を使い、`k=10` で次の優先順位で判定する。

1. `rank_by_mora_editdistance` で recall@10=1 の query を `easy`
2. それ以外で `rank_by_vowel_consonant_editdistance(vowel_ratio=0.8)` で recall@10=1 の query を `medium`
3. どちらでも recall@10=1 にならない query を `hard`

## Decision

- 各 query オブジェクトに `difficulty` フィールドを追加して難易度ラベルを直接保持する
- ルート `metadata` に難易度判定ルール（`k`、優先順位、ルール文）を保持する

## Consequences

- 利用側は query 単位で難易度フィルタを直接書けるため、サブセット評価が簡単になる
- ルールが `metadata` に残るため再現性が維持される
