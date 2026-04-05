# -*- coding: utf-8 -*-
"""
FAISSインデックス生成スクリプト
chunker.py が出力したJSONファイルからベクトルインデックスを生成する

使用例:
    python make_index.py -i merosu.json -o index_dir
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "./models/embedded/paraphrase-multilingual-MiniLM-L12-v2"

# 出力ファイル名
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE    = "metadata.pkl"


def load_chunks(input_path: str) -> list[dict]:
    """チャンクJSONを読み込む"""
    with open(input_path, encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        print("エラー: 入力JSONが空です", file=sys.stderr)
        sys.exit(1)
    return chunks


def build_index(chunks: list[dict], model: SentenceTransformer) -> tuple[faiss.Index, list[dict]]:
    """チャンクをエンベディングしてFAISSインデックスを構築する"""
    texts = [c["text"] for c in chunks]

    print(f"エンベディング生成中... ({len(texts)} チャンク)", file=sys.stderr)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # L2正規化してコサイン類似度検索に対応
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内積（正規化済みなのでコサイン類似度と同等）
    index.add(embeddings)

    print(f"インデックス構築完了: {index.ntotal} ベクトル, 次元数={dim}", file=sys.stderr)
    return index, chunks


def save_index(index: faiss.Index, metadata: list[dict], output_dir: str) -> None:
    """インデックスとメタデータを保存する"""
    os.makedirs(output_dir, exist_ok=True)

    index_path    = os.path.join(output_dir, FAISS_INDEX_FILE)
    metadata_path = os.path.join(output_dir, METADATA_FILE)

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"保存完了:", file=sys.stderr)
    print(f"  FAISSインデックス : {index_path}", file=sys.stderr)
    print(f"  メタデータ        : {metadata_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="JSONチャンクからFAISSベクトルインデックスを生成します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python make_index.py -i merosu.json -o index_dir
        """,
    )
    parser.add_argument("-i", "--input",      required=True, help="入力JSONファイルパス（chunker.py の出力）")
    parser.add_argument("-o", "--output_dir", required=True, help="インデックス出力ディレクトリ")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"エラー: 入力ファイルが見つかりません: {args.input}", file=sys.stderr)
        sys.exit(1)

    # チャンク読み込み
    chunks = load_chunks(args.input)
    print(f"チャンク数: {len(chunks)}", file=sys.stderr)

    # モデルロード
    print(f"モデルロード中: {MODEL_NAME}", file=sys.stderr)
    model = SentenceTransformer(MODEL_NAME)

    # インデックス構築
    index, metadata = build_index(chunks, model)

    # 保存
    save_index(index, metadata, args.output_dir)


if __name__ == "__main__":
    main()
