# -*- coding: utf-8 -*-
"""
RAG CLI
5_search.py で類似チャンクを取得し 6_generate.py で回答を生成する

使用例:
    python 7_rag_cli.py -q "メロスはなぜ怒ったのか"
    python 7_rag_cli.py -i chunk -q "メロスはなぜ怒ったのか"
"""

from __future__ import annotations

import argparse
import os
import sys

from importlib import import_module

_search_mod   = import_module("5_search")
load_index    = _search_mod.load_index
search        = _search_mod.search

_generate_mod = import_module("6_generate")
generate      = _generate_mod.generate

from sentence_transformers import SentenceTransformer

# 固定値
EMBED_MODEL = "./models/embedded/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 5
N_PREDICT   = 512
CTX_SIZE    = 2048


def main():
    parser = argparse.ArgumentParser(
        description="RAG: ベクトル検索 + llama.cpp による回答生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python rag_cli.py -q "メロスはなぜ怒ったのか"
  python rag_cli.py -i chunk -q "メロスはなぜ怒ったのか"
        """,
    )
    parser.add_argument("-m", "--model", default="./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
                        help="LLM モデルファイルパス (.gguf)（デフォルト: ./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf）")
    parser.add_argument("-i", "--index_dir", default="index_dir",
                        help="インデックスディレクトリ（デフォルト: index_dir）")
    parser.add_argument("-q", "--query", required=True,
                        help="クエリ文字列")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"エラー: モデルファイルが見つかりません: {args.model}", file=sys.stderr)
        sys.exit(1)

    # インデックス読み込み
    index, metadata = load_index(args.index_dir)

    # 埋め込みモデルロード
    model = SentenceTransformer(EMBED_MODEL)

    # 検索
    chunks = search(args.query, index, metadata, model, TOP_K)

    if not chunks:
        print("エラー: 検索結果が空です", file=sys.stderr)
        sys.exit(1)

    # 検索結果のサマリを stderr に出力
    print("--- 検索結果サマリ ---", file=sys.stderr)
    for c in chunks:
        preview = c["text"][:40].replace("\n", " ")
        print(f"  id={c['id']}  score={c['score']:.4f}  tokens={c['token_count']}  \"{preview}...\"", file=sys.stderr)
    print("---------------------", file=sys.stderr)

    # 回答生成
    answer = generate(
        chunks     = chunks,
        query      = args.query,
        model_path = args.model,
        n_predict  = N_PREDICT,
        ctx_size   = CTX_SIZE,
    )

    print(answer)


if __name__ == "__main__":
    main()
