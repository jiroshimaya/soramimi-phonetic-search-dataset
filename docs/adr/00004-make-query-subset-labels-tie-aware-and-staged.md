# ADR 00004: Make query subset labels tie-aware and staged

- Status: accepted
- Date: 2026-06-21
- Supersedes: 00003-store-query-subset-labels.md
- Superseded by: none

## Context

`baseball.json` の `easy` / `medium` / `hard` は、これまで
`rank_by_mora_editdistance` と
`rank_by_vowel_consonant_editdistance(vowel_ratio=0.8)` の
recall@10=1 判定だけで付与していた。

ただし実装上は同点候補の二次キーを持っておらず、同一スコアの候補が多い query では
元の `wordlist` 順に依存して top 10 に正解が滑り込むことがあった。
そのため、tie-break に依存して解けている query が実際より易しく見える問題があった。

また、`easy_tie` のように mora では不安定でも、
次段の vowel/consonant では安定して解ける query と、
どちらでも安定して解けない query は分けて扱いたい。

## Decision

- 難易度ラベルは段階的に付与する
  1. `rank_by_mora_editdistance` で、同点を負例優先で崩しても recall@10=1 を維持できる query を `easy`
  2. それ以外で `rank_by_vowel_consonant_editdistance(vowel_ratio=0.8)` で、同じ pessimistic tie-break を適用しても recall@10=1 を維持できる query を `medium`
  3. どちらでも安定して recall@10=1 を満たさない query を `hard`
- `baseball.json` の各 query に再計算した `subset` を保存する
- ルート `metadata.subset_definition` に staged 判定と tie-break 方針を保存する
- このラベル再生成と、保存済み結果 JSON の `recall_by_subset` 更新に使うコードを `scripts/update_subset_labels.py` と `src/soramimi_phonetic_search_dataset/subset_labeling.py` に保存する

## Consequences

- `easy` / `medium` / `hard` が、元の語彙順ではなく「そのベースラインで安定して解けるか」を表すようになる
- 従来 `easy` だった一部 query は `medium` または `hard` へ再分類され、`medium` の意味が明確になる
- subset ラベル変更時に結果 JSON の `recall_by_subset` も更新する運用が必要になる
