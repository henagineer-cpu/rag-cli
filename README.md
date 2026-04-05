# ローカルRAGパイプライン

GPUなし・クラウドなしで動作する、完全ローカルの RAG（Retrieval-Augmented Generation）実装です。
テキストをベクトル検索して関連箇所を取り出し、その内容を LLM への文脈として渡すことで回答を生成します。

## 動作環境

| 項目 | 内容 |
|---|---|
| OS | Linux / macOS（WSL2 動作確認済み） |
| Python | 3.14（conda 管理） |
| LLM 推論 | llama.cpp（別途ビルド必要） |
| 埋め込みモデル | paraphrase-multilingual-MiniLM-L12-v2 |

## ファイル構成

```
0_setup.sh              # Miniconda インストール＋conda 環境構築
1_pip_install.sh        # pip パッケージ一括インストール
2_get_embedded_model.sh # 埋め込みモデルのダウンロード（初回のみ）
3_make_chunk.py         # テキスト → チャンク JSON
4_make_index.py         # チャンク JSON → FAISS インデックス
5_search.py             # クエリ → 類似チャンク取得
6_generate.py           # チャンク + クエリ → llama.cpp → 回答生成
7_rag_cli.py            # 5_search + 6_generate をまとめた CLI
environment.yml         # conda 環境定義
requirements.txt        # pip 依存パッケージ
merosu.txt              # サンプルテキスト（走れメロス）
```

## セットアップ

llama-cliにPATHが通っていることを確認してください。
後は以下の手順通りに実行すれがよいです。


### 3. Python 環境の構築

```bash
# conda が未インストールの場合は自動インストール
bash 0_setup.sh
source ~/.bashrc
conda activate rag-env

# Python パッケージのインストール
bash 1_pip_install.sh
```

### 4. 埋め込みモデルのダウンロード（初回のみ）

```bash
bash 2_get_embedded_model.sh
```

`models/embedded/paraphrase-multilingual-MiniLM-L12-v2/` にダウンロードされます。



## 使い方

パイプライン全体の流れ：

```
テキスト
  ↓ 3_make_chunk.py
チャンク JSON
  ↓ 4_make_index.py
FAISS インデックス
  ↓ 7_rag_cli.py（クエリ入力）
回答
```

### ステップ 1: テキストのチャンク分割

```bash
python 3_make_chunk.py -i merosu.txt -o merosu.json -c 100 -s 50
```

| オプション | 説明 | デフォルト |
|---|---|---|
| `-i` | 入力テキストファイル | 必須 |
| `-o` | 出力 JSON ファイル | 必須 |
| `-c` | チャンクサイズ（文字数） | 512 |
| `-s` | オーバーラップサイズ（文字数） | 64 |

### ステップ 2: FAISS インデックスの生成

```bash
python 4_make_index.py -i merosu.json -o index_dir
```

`index_dir/` に `faiss.index` と `metadata.pkl` が生成されます。

### ステップ 3: RAG 回答生成

```bash
python 7_rag_cli.py -m ./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf -q "メロスはなぜ怒ったのか"
```

| オプション | 説明 | デフォルト |
|---|---|---|
| `-q` | クエリ文字列 | 必須 |
| `-m` | LLM モデルファイルパス (.gguf) | `./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf` |
| `-i` | インデックスディレクトリ | `index_dir` |

#### 実行例

```
$ python 7_rag_cli.py -m ./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf -q "メロスはなぜ怒ったのか"

--- 検索結果サマリ ---

> 以下の文脈情報を参考にして、質問に答えてください。
質問: メロスはなぜ怒ったのか

### 文脈
[1] ゅんらの警吏に捕縛された。調べられて、メロスの懐中からは短剣が出て来たので、騒ぎが大きくなってしまった。メロスは、王の前に引き出された。「この短刀で何をするつもりであったか。言え！」暴君ディオニスは静


メロスは、王の前に引き出され、王から短剣がもたらされたことにより、彼の行動や将来の運命に関する疑問を抱いたため怒った。
[ Prompt: 86.1 t/s | Generation: 23.5 t/s ]

```

### 各スクリプトを個別に使う場合

```bash
# 検索のみ
python 5_search.py -i index_dir -q "メロスはなぜ怒ったのか" -k 5

# 検索結果 JSON から回答生成
python 6_generate.py -i result.json -q "メロスはなぜ怒ったのか" \
    -m ./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
```

## 設定変更

`7_rag_cli.py` 冒頭の定数を編集することで、検索件数や生成トークン数を変更できます。モデルパスは `-m` オプションで指定してください。

```python
EMBED_MODEL = "./models/embedded/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 5
N_PREDICT   = 512
CTX_SIZE    = 2048
```

## 依存パッケージ

| パッケージ | 役割 |
|---|---|
| `faiss-cpu` | ベクトルインデックス・検索 |
| `sentence-transformers` | テキストの埋め込みベクトル生成 |
| `numpy` | ベクトル演算 |
| `huggingface_hub` | 埋め込みモデルのダウンロード |


