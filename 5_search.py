# -*- coding: utf-8 -*-
"""
FAISS ベクトル検索スクリプト
make_index.py が生成したインデックスに対してクエリ検索を行い、結果をJSON形式で標準出力する

使用例:
    python search.py -i index_dir -q "メロスはなぜ怒ったのか" -k 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "./models/embedded/paraphrase-multilingual-MiniLM-L12-v2"

FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE    = "metadata.pkl"


def load_index(index_dir: str) -> tuple[faiss.Index, list[dict]]:
    """FAISSインデックスとメタデータを読み込む"""
    index_path    = os.path.join(index_dir, FAISS_INDEX_FILE)
    metadata_path = os.path.join(index_dir, METADATA_FILE)

    if not os.path.isfile(index_path):
        print(f"エラー: インデックスファイルが見つかりません: {index_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(metadata_path):
        print(f"エラー: メタデータファイルが見つかりません: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def search(query: str, index: faiss.Index, metadata: list[dict],
           model: SentenceTransformer, top_k: int) -> list[dict]:
    """クエリをエンベディングしてFAISSで類似チャンクを検索する"""
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = metadata[idx]
        results.append({
            "id":          chunk["id"],
            "text":        chunk["text"],
            "source":      chunk["source"],
            "token_count": chunk["token_count"],
            "score":       float(score),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="FAISSインデックスに対してベクトル検索を行いJSON形式で出力します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python search.py -i index_dir -q "メロスはなぜ怒ったのか" -k 5
        """,
    )
    parser.add_argument("-i", "--index_dir", required=True, help="インデックスディレクトリ（make_index.py の出力先）")
    parser.add_argument("-q", "--query",     required=True, help="検索クエリ")
    parser.add_argument("-k", "--top_k",     type=int, default=5, help="取得する上位件数（デフォルト: 5）")
    args = parser.parse_args()

    if args.top_k <= 0:
        print("エラー: top_k は1以上の整数で指定してください", file=sys.stderr)
        sys.exit(1)

    # インデックス読み込み
    index, metadata = load_index(args.index_dir)

    # モデルロード
    model = SentenceTransformer(MODEL_NAME)

    # 検索
    results = search(args.query, index, metadata, model, args.top_k)

    # JSON出力（標準出力）
    output = {
        "query":   args.query,
        "results": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
