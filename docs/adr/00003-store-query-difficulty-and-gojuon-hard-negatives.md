# ADR 00003: Store query difficulty and gojuon hard negatives

- Status: accepted
- Date: 2026-06-04
- Supersedes: none
- Superseded by: none

## Context

`baseball.json` に難易度サブセットを導入したい。難易度は `easy` / `medium` / `hard` を使い、`k=10` で次の優先順位で判定する。

1. `rank_by_mora_editdistance` で recall@10=1 の query を `easy`
2. それ以外で `rank_by_vowel_consonant_editdistance(vowel_ratio=0.8)` で recall@10=1 の query を `medium`
3. どちらでも recall@10=1 にならない query を `hard`

また、`hard_negatives` は候補選定の意味と表示上の並びを分離したい。候補選定は従来どおり音韻距離ベースで行いつつ、公開データ上は五十音順で読みやすくしたい。

## Decision

- 各 query オブジェクトに `difficulty` フィールドを追加して難易度ラベルを直接保持する
- ルート `metadata` に難易度判定ルール（`k`、優先順位、ルール文）を保持する
- `hard_negatives` は生成元を `rank_by_vowel_consonant_editdistance(vowel_ratio=0.5)` の上位100件（positive除外）とし、公開JSONでは五十音順で保存する

## Consequences

- 利用側は query 単位で難易度フィルタを直接書けるため、サブセット評価が簡単になる
- ルールが `metadata` に残るため再現性が維持される
- `hard_negatives` の表示順が距離順ではなくなるため、順位情報が必要な用途では再計算が必要になる
