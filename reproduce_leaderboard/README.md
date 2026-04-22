# Reproduce Leaderboard

soramimi-phonetic-search-datasetのleaderboardの再現用スクリプト群です。

## 概要

[leaderboard.md](../leaderboard.md)に記載されている各手法のRecall@10を再現するためのスクリプトです。

軽量な試行錯誤用として、先頭10クエリだけで評価した [`leaderboard.small.md`](./leaderboard.small.md) も用意しています。こちらは開発・研究時のスモークテスト向けで、トップレベル README からはリンクしていません。

## インストール

1. プロジェクトのルートディレクトリで以下のコマンドを実行してパッケージをインストールします：

```bash
uv pip install -e .
```

2. 評価用の依存関係をインストールします：

```bash
uv pip install --group evaluation
```

## 使い方

### 全ての手法を実行

```bash
sh run_all.sh
```

### 個別の手法を実行

```bash
uv run methods/000_mora.py  # モーラ編集距離
uv run methods/001_phoneme.py  # 音素編集距離
uv run methods/002_vowel_consonant.py  # 母音子音編集距離
uv run methods/003_kanasim.py  # KanaSim編集距離
uv run methods/004_llm_rerank_gpt4o_mini.py  # LLMリランク (gpt-4o-mini)
uv run methods/005_llm_rerank_gpt4o.py  # LLMリランク (gpt-4o)
uv run methods/006_llm_rerank_gemini.py  # LLMリランク (gemini-2.0-flash)
uv run methods/007_llm_rerank_gpt45preview.py  # LLMリランク (gpt-4.5-preview)
uv run methods/008_llm_rerank_gpt54.py  # LLMリランク (gpt-5.4)
uv run methods/008_01_llm_rerank_gpt54_simple.py  # LLMリランク (gpt-5.4, prompt 008_01 simple)
uv run methods/008_02_llm_rerank_gpt54_detailed.py  # LLMリランク (gpt-5.4, prompt 008_02 detailed)
uv run methods/008_03_llm_rerank_gpt54_step_by_step.py  # LLMリランク (gpt-5.4, prompt 008_03 step-by-step)
uv run methods/008_04_llm_rerank_gpt54_detailed_pyopenjtalk_romaji.py  # LLMリランク (gpt-5.4, prompt 008_02 detailed, pyopenjtalk romaji input)
uv run methods/010_01_llm_rerank_gpt54_medium_simple.py  # LLMリランク (gpt-5.4, reasoning medium, prompt 010_01 simple)
uv run methods/010_02_llm_rerank_gpt54_medium_detailed.py  # LLMリランク (gpt-5.4, reasoning medium, prompt 010_02 detailed)
uv run methods/010_03_llm_rerank_gpt54_medium_step_by_step.py  # LLMリランク (gpt-5.4, reasoning medium, prompt 010_03 step-by-step)
```

### カスタム評価の実行

カスタムパラメータでの評価を行う場合は、以下のスクリプトを使用できます：

```bash
# ヘルプを表示
uv run methods/common/evaluate_ranking.py --help

# 母音子音編集距離でtop10を評価
uv run methods/common/evaluate_ranking.py -r vowel_consonant -n 10

# 先頭10クエリだけで軽く試す
uv run methods/common/evaluate_ranking.py -r vowel_consonant -n 10 --dataset_size small

# 母音の重みを変更（kanasim, vowel_consonantの場合のみ有効）
uv run methods/common/evaluate_ranking.py -r vowel_consonant -vr 0.7

# KanaSimとLLMリランクを組み合わせて評価
uv run methods/common/evaluate_ranking.py -r kanasim --rerank --rerank_model_name gpt-4o-mini

# GPT-5.4でLLMリランク
uv run methods/common/evaluate_ranking.py -r vowel_consonant --rerank --rerank_model_name gpt-5.4

# GPT-5.4で reasoning effort none と prompt variant を指定
uv run methods/common/evaluate_ranking.py -r vowel_consonant --rerank --rerank_model_name gpt-5.4 --rerank_reasoning_effort none --rerank_prompt_template 008_02_detailed

# GPT-5.4で pyopenjtalk ローマ字入力を使って rerank
uv run methods/common/evaluate_ranking.py -r vowel_consonant --rerank --rerank_model_name gpt-5.4 --rerank_reasoning_effort none --rerank_prompt_template 008_02_detailed --rerank_input_transform pyopenjtalk_romaji

# 評価結果の保存先を指定
uv run methods/common/evaluate_ranking.py -o output.json

# 評価結果を保存しない
uv run methods/common/evaluate_ranking.py --no_save
```

#### オプション

- `-r`, `--rank_func`: ランキング関数の種類（kanasim, vowel_consonant, phoneme, mora）
- `-n`, `--topn`: 評価に使用する上位n件
- `--dataset_size`: データセットサイズ（default, small）
- `-vr`, `--vowel_ratio`: 母音の重み（kanasim, vowel_consonantの場合のみ使用）
- `--rerank`: LLMによるリランキングを使用
- `--rerank_input_size`: リランクに使用する候補数
- `--rerank_batch_size`: リランクのバッチサイズ
- `--rerank_model_name`: リランクに使用するモデル名
- `--rerank_reasoning_effort`: 対応モデルで使用する reasoning effort（none, low, medium, high）
- `--rerank_prompt_template`: LLMリランクに使う system prompt（default, 008_01_simple, 008_02_detailed, 008_03_step_by_step）
- `--rerank_input_transform`: LLMへ渡す前の query / candidate 変換（none, pyopenjtalk_romaji）
- `--rerank_interval`: リランクのインターバル（秒）
- `-o`, `--output_file_path`: 出力ファイルのパス
- `--no_save`: 評価結果を保存しない

## 結果の出力

各手法の実行結果は`results/`ディレクトリに保存されます：

```
results/
├── 000_mora.json
├── 001_phoneme.json
├── 002_vowel_consonant.json
├── 003_kanasim.json
├── 004_llm_rerank_gpt4o_mini.json
├── 005_llm_rerank_gpt4o.json
├── 006_llm_rerank_gemini.json
├── 007_llm_rerank_gpt45preview.json
├── 008_llm_rerank_gpt54.json
├── 008_01_llm_rerank_gpt54_simple.json
├── 008_02_llm_rerank_gpt54_detailed.json
├── 008_03_llm_rerank_gpt54_step_by_step.json
├── 008_04_llm_rerank_gpt54_detailed_pyopenjtalk_romaji.json
├── 010_01_llm_rerank_gpt54_medium_simple.json
├── 010_01_llm_rerank_gpt54_medium_simple_cost_estimate.json
├── 010_02_llm_rerank_gpt54_medium_detailed.json
├── 010_03_llm_rerank_gpt54_medium_step_by_step.json
└── 010_03_llm_rerank_gpt54_medium_step_by_step_cost_estimate.json
```

`*_cost_estimate.json` は、先頭10件で計測した token/cost を全150件へ線形外挿した**試算**です。full run の実測値ではありません。

small dataset を使って試した結果は、`results_small/` に保存すると整理しやすいです。リポジトリには small 実測のサンプルとして `leaderboard.small.md` に対応する JSON を配置しています。

## 注意事項

- 評価には`baseball.json`データセットが使用されます。
- `--dataset_size small` を使うと、同じ単語リストのまま先頭10クエリだけで評価できます。
- 各ランキング関数のパラメータは必要に応じて調整できます。
- LLMリランクを使用する場合は、以下の環境変数を設定してください：
  - OpenAI API（gpt-4o-mini, gpt-4o, gpt-4.5-preview, gpt-5.4）を使用する場合：
    - `OPENAI_API_KEY`: OpenAIのAPIキー
  - Gemini API（gemini-2.0-flash）を使用する場合：
    - `GEMINI_API_KEY`: Google Cloud PlatformのAPIキー

環境変数は以下のいずれかの方法で設定できます：

1. シェルで直接設定：
```bash
export OPENAI_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-api-key"
```

2. `.env`ファイルを作成：
```bash
# .envファイルの例
OPENAI_API_KEY=your-api-key
GOOGLE_API_KEY=your-api-key
```
