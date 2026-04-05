# はじめに

前回の記事ではローカルLLMをCPUだけで動かす方法を紹介しました。今回はその発展として、**RAG（Retrieval-Augmented Generation）** を実装してみます。RAGとは、あらかじめ用意したテキストをベクトル検索して関連箇所を取り出し、その内容をLLMへの文脈として渡すことで、より正確な回答を得る手法です。GPUなし・クラウドなし・すべてローカルで動かします。


# 対象者
@@変更不要
GPUなし・クラウドなしのローカル環境でRAGを試したい人。
今回も10年ほど前のデスクトップのWSLで動かしています。

## 環境
@@変更不要
OS       : Ubuntu 24.04.1 LTS  
CPU      : Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz  
メモリ    : 16GB  
llama.cpp: version 8146 (418dea39c)




## 関連記事
@@変更不要
[CPUだけでローカルLLMを動かしてみた](https://qiita.com/henagineer/items/3b167e5a028741b61e22)  

## 開発環境

Python の仮想環境には **conda** を使用しています。

```bash
conda create -n rag-env python=3.14
conda activate rag-env
pip install -r requirements.txt
```

主な依存パッケージ（`requirements.txt`）：

| パッケージ | 役割 |
|---|---|
| `faiss-cpu` | ベクトルインデックス・検索 |
| `sentence-transformers` | テキストの埋め込みベクトル生成 |
| `numpy` | ベクトル演算 |
| `huggingface_hub` | 埋め込みモデルのダウンロード |

LLM の推論には別途ビルドした **llama.cpp** を使用します。

## 全体の構造

今回実装したファイルは番号付きで実行順序が分かるようにしています。

```
0_setup.sh                  # Miniconda インストール＋conda 環境構築
1_pip_install.sh            # pip パッケージ一括インストール
2_get_embedded_model.sh     # 埋め込みモデルのダウンロード（初回のみ）
3_make_chunk.py             # テキスト → チャンク JSON
4_make_index.py             # チャンク JSON → FAISS インデックス
5_search.py                 # クエリ → 類似チャンク取得
6_generate.py               # チャンク + クエリ → llama.cpp → 回答生成
7_rag_cli.py                # 5_search + 6_generate をまとめた CLI
```

パイプライン全体の流れ：

```
テキスト
  ↓ 3_make_chunk.py
チャンク JSON
  ↓ 4_make_index.py
FAISS インデックス
  ↓ 5_search.py（クエリ入力）
類似チャンク
  ↓ 6_generate.py
回答
```

`7_rag_cli.py` は `5_search.py` と `6_generate.py` を import して一度に実行するラッパーです。

## 各スクリプトの機能と使い方

題材として青空文庫の「走れメロス」を使います。

### 3_make_chunk.py ― テキストをチャンクに分割する

長いテキストをベクトル検索に適したサイズに分割してJSON形式で保存します。

```bash
python 3_make_chunk.py -i merosu.txt -o chunk/merosu.json -c 100 -s 50
```

| オプション | 説明 |
|---|---|
| `-i` | 入力テキストファイル |
| `-o` | 出力JSONファイル |
| `-c` | チャンクサイズ（文字数） |
| `-s` | オーバーラップサイズ（文字数） |

出力JSONのフォーマット：
```json
[
  {
    "id": 0,
    "text": "メロスは激怒した。必ず、かの邪智暴虐の王を...",
    "source": "merosu.txt",
    "token_count": 98
  }
]
```

### 4_make_index.py ― FAISSインデックスを生成する

チャンクJSONを読み込み、埋め込みモデル（`paraphrase-multilingual-MiniLM-L12-v2`）でベクトル化してFAISSインデックスを生成します。

```bash
python 4_make_index.py -i chunk/merosu.json -o index_dir
```

`index_dir/` に以下の2ファイルが生成されます：
- `faiss.index` … ベクトルの実体
- `metadata.pkl` … id / text / source / token_count

### 5_search.py ― クエリで類似チャンクを検索する

クエリをベクトル化してFAISSで類似チャンクを取得しJSON形式で出力します。

```bash
python 5_search.py -i index_dir -q "メロスはなぜ怒ったのか" -k 5
```

出力例：
```json
{
  "query": "メロスはなぜ怒ったのか",
  "results": [
    {
      "id": 0,
      "text": "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと...",
      "score": 0.8821
    }
  ]
}
```

### 6_generate.py ― llama.cpp で回答を生成する

検索結果チャンクとクエリからプロンプトを組み立て、llama-cli に渡して回答を生成します。

プロンプトのテンプレート：
```
以下の文脈情報を参考にして、質問に答えてください。
質問: メロスはなぜ怒ったのか

### 文脈
[1] メロスは激怒した。必ず、かの邪智暴虐の王を...
[2] ...

### 質問
メロスはなぜ怒ったのか

### 回答
```

単体で使う場合：
```bash
python 6_generate.py -i result.json -q "メロスはなぜ怒ったのか" -m ./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
```

### 7_rag_cli.py ― 検索から回答生成まで一括実行する

検索→回答生成をまとめて実行するCLIです。

```bash
python 7_rag_cli.py -q "メロスはなぜ怒ったのか"
```

実行すると検索結果のサマリが表示されたあと、回答が出力されます：

```
--- 検索結果サマリ ---
  id=0  score=0.8821  tokens=98  "メロスは激怒した。必ず、かの邪智暴虐の王を..."
  id=3  score=0.7643  tokens=102  "王は静かに、しかし威厳をもって言った..."
---------------------
メロスは、邪智暴虐の王に対して怒りを覚えました。...
```

内部の固定値（モデルパス・検索件数など）は `7_rag_cli.py` の先頭で変更できます：

```python
MODEL_PATH  = "./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
EMBED_MODEL = "./models/embedded/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 5
N_PREDICT   = 512
CTX_SIZE    = 2048
```


# まとめ
