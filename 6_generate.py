# -*- coding: utf-8 -*-
"""
llama.cpp を使った回答生成スクリプト
search.py の JSON出力を受け取り、プロンプトを組み立てて llama-cli に渡し回答を生成する

使用例:
    python generate.py -i results.json -q "メロスはなぜ怒ったのか" -m ./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

LLAMA_CLI = "llama-cli"

DEFAULT_N_PREDICT = 512
DEFAULT_CTX_SIZE  = 2048


def build_prompt(chunks: list[dict], query: str) -> str:
    """検索結果チャンクとクエリからプロンプトを組み立てる"""
    context = "\n\n".join(
        f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(chunks)
    )
    return (
        f"以下の文脈情報を参考にして、質問に答えてください。\n質問: {query}\n\n"
        f"### 文脈\n{context}\n\n"
        f"### 質問\n{query}\n\n"
        "### 回答\n"
    )


def generate(chunks: list[dict], query: str, model_path: str,
             n_predict: int = DEFAULT_N_PREDICT,
             ctx_size: int = DEFAULT_CTX_SIZE) -> str:
    """プロンプトを組み立てて llama-cli に渡し回答文字列を返す"""
    prompt = build_prompt(chunks, query)
    print("--- 送信プロンプト ---", file=sys.stderr)
    print(prompt, file=sys.stderr)
    print("---------------------", file=sys.stderr)

    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "-c", str(ctx_size),
        "--no-display-prompt",
        "-e",
        "--log-disable",
        "--log-verbosity", "-1",
        "--single-turn",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print(f"エラー: llama-cli が見つかりません: {LLAMA_CLI}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"エラー: llama-cli の実行に失敗しました\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

    return result.stdout.strip()


def load_chunks(input_arg: str) -> tuple[str, list[dict]]:
    """ファイルから search.py の JSON を読み込む"""
    if not os.path.isfile(input_arg):
        print(f"エラー: 入力ファイルが見つかりません: {input_arg}", file=sys.stderr)
        sys.exit(1)
    with open(input_arg, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"エラー: JSONの解析に失敗しました: {e}", file=sys.stderr)
            sys.exit(1)

    chunks = data.get("results", [])
    query  = data.get("query", "")
    return query, chunks


def main():
    parser = argparse.ArgumentParser(
        description="search.py の検索結果を受け取り llama.cpp で回答を生成します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python generate.py -i answer.json -q "メロスはなぜ怒ったのか" -m ./models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
        """,
    )
    parser.add_argument("-i", "--input",     required=True,
                        help="search.py の JSON出力ファイルパス")
    parser.add_argument("-q", "--query",     required=True,
                        help="クエリ文字列")
    parser.add_argument("-m", "--model",     required=True,
                        help="llama.cpp モデルパス（.gguf）")
    parser.add_argument("-n", "--n-predict", type=int, default=DEFAULT_N_PREDICT,
                        help=f"最大生成トークン数（デフォルト: {DEFAULT_N_PREDICT}）")
    parser.add_argument("-c", "--ctx-size",  type=int, default=DEFAULT_CTX_SIZE,
                        help=f"コンテキストサイズ（デフォルト: {DEFAULT_CTX_SIZE}）")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"エラー: モデルファイルが見つかりません: {args.model}", file=sys.stderr)
        sys.exit(1)

    _, chunks = load_chunks(args.input)

    if not chunks:
        print("エラー: 検索結果チャンクが空です", file=sys.stderr)
        sys.exit(1)

    answer = generate(
        chunks      = chunks,
        query       = args.query,
        model_path  = args.model,
        n_predict   = args.n_predict,
        ctx_size    = args.ctx_size,
    )

    print(answer)


if __name__ == "__main__":
    main()
