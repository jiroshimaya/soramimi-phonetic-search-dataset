# Examples

このディレクトリには、soramimi-phonetic-search-datasetの使用例が含まれています。

## basic_usage.py

基本的な使用例を示すスクリプトです。以下の機能を含みます：

- 各種ランキング関数（kanasim, vowel_consonant, phoneme, mora）の評価
- LLMを使用したリランキング
- 評価結果のJSON形式での保存

### 使用方法

```bash
# 基本的な使用方法（母音子音編集距離を使用）
python examples/basic_usage.py

# 異なるランキング関数を使用
python examples/basic_usage.py -r kanasim
python examples/basic_usage.py -r phoneme
python examples/basic_usage.py -r mora

# 母音の重みを変更（kanasim, vowel_consonantの場合のみ有効）
python examples/basic_usage.py -r vowel_consonant -vr 0.7

# LLMによるリランキングを使用
python examples/basic_usage.py --rerank --rerank_model_name gpt-4o-mini

# 評価結果の保存先を指定
python examples/basic_usage.py -o output.json

# 評価結果を保存しない
python examples/basic_usage.py --no_save
```

### オプション

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

## 実行方法

1. まず、プロジェクトのルートディレクトリで以下のコマンドを実行してパッケージをインストールします：

```bash
pip install -e .
```

2. サンプルコードを実行：

```bash
# 基本的な使用例
python examples/basic_usage.py

# パラメータをカスタマイズした例
python examples/basic_usage.py -r vowel_consonant -vr 0.7 -n 5
```

## 注意事項

- サンプルコードを実行する前に、必要なパッケージがすべてインストールされていることを確認してください。
- 各ランキング関数のパラメータは必要に応じて調整できます。
- 評価には`baseball.json`データセットが使用されます。 