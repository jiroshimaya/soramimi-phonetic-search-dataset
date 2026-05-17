# ADR 00002: Separate fine-grained experiments from the main dataset repository

- Status: accepted
- Date: 2026-05-17
- Supersedes: none
- Superseded by: none

## Context

`soramimi-phonetic-search-dataset` では leaderboard 再現用のスクリプトと結果を同居させてきたが、prompt 差分や input variation、probe のような細かな試行錯誤まで本家リポジトリに残すと、公開面の説明コストが上がり、代表的な結果が埋もれやすくなっていた。

一方で、削除した実験コードと結果も、後から比較や再現ができる形では保持したい。

## Decision

代表的な手法と結果だけを `soramimi-phonetic-search-dataset` に残し、細かな派生実験は別リポジトリ `soramimi-phonetic-search-experiments` で管理する。

本家に残すのは、公開 leaderboard に載せる代表実装とその結果に限る。
prompt/input の派生比較、small dataset の追加試行、structured outputs の probe、別モデルでの検証は experiment リポジトリへ移す。

## Consequences

本家リポジトリの leaderboard と再現手順は短く保てる。

細かな比較や途中経過も experiment リポジトリに残るため、再現性と調査可能性は維持できる。

今後は細かな検証を追加するとき、本家へ直接ファイルを増やすのではなく experiment リポジトリ側へ追加する運用が必要になる。
