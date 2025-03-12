# Reproduce Leaderboard

soramimi-phonetic-search-datasetのleaderboardの再現用スクリプト群です。

## 概要

[leaderboard.md](../leaderboard.md)に記載されている各手法のRecall@10を再現するためのスクリプトです。

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
```

### カスタム評価の実行

カスタムパラメータでの評価を行う場合は、以下のスクリプトを使用できます：

```bash
# ヘルプを表示
uv run methods/common/evaluate_ranking.py --help

# 母音子音編集距離でtop10を評価
uv run methods/common/evaluate_ranking.py -r vowel_consonant -n 10

# 母音の重みを変更（kanasim, vowel_consonantの場合のみ有効）
uv run methods/common/evaluate_ranking.py -r vowel_consonant -vr 0.7

# KanaSimとLLMリランクを組み合わせて評価
uv run methods/common/evaluate_ranking.py -r kanasim --rerank --rerank_model_name gpt-4o-mini

# 評価結果の保存先を指定
uv run methods/common/evaluate_ranking.py -o output.json

# 評価結果を保存しない
uv run methods/common/evaluate_ranking.py --no_save
```

#### オプション

- `-r`, `--rank_func`: ランキング関数の種類（kanasim, vowel_consonant, phoneme, mora）
- `-n`, `--topn`: 評価に使用する上位n件
- `-vr`, `--vowel_ratio`: 母音の重み（kanasim, vowel_consonantの場合のみ使用）
- `--rerank`: LLMによるリランキングを使用
- `--rerank_input_size`: リランクに使用する候補数
- `--rerank_batch_size`: リランクのバッチサイズ
- `--rerank_model_name`: リランクに使用するモデル名
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
└── 007_llm_rerank_gpt45preview.json
```

## 注意事項

- 評価には`baseball.json`データセットが使用されます。
- 各ランキング関数のパラメータは必要に応じて調整できます。
- LLMリランクを使用する場合は、以下の環境変数を設定してください：
  - OpenAI API（gpt-4o-mini, gpt-4o, gpt-4.5-preview）を使用する場合：
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