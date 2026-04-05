# -*- coding: utf-8 -*-
"""
テキストチャンカー
テキストファイルを指定サイズにチャンクしてJSON形式で出力する

使用例:
    python chunker.py -i input.txt -o output.json -c 512 -s 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys


import re


def _is_japanese(text: str) -> bool:
    """日本語文字（ひらがな・カタカナ・漢字）を含むか判定"""
    return bool(re.search(r'[\u3040-\u9FFF\uF900-\uFAFF]', text))


def _tokenize(text: str) -> list[str]:
    """テキストをトークンに分割する。
    日本語: 1文字1トークン（空白・改行は除外）
    英語 : 空白区切り
    """
    if _is_japanese(text):
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def _join_tokens(tokens: list[str], is_japanese: bool) -> str:
    """トークンリストをテキストに結合する"""
    if is_japanese:
        return "".join(tokens)
    return " ".join(tokens)


def count_tokens(text: str) -> int:
    """トークン数を返す"""
    return len(_tokenize(text))


def chunk_text(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    """
    テキストをトークン単位でチャンクに分割する

    Args:
        text: 入力テキスト
        chunk_size: チャンクサイズ（トークン数）
        overlap_size: オーバーラップサイズ（トークン数）

    Returns:
        チャンクのリスト
    """
    is_jp = _is_japanese(text)
    tokens = _tokenize(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunks.append(_join_tokens(tokens[start:end], is_jp))

        if end >= len(tokens):
            break

        start += chunk_size - overlap_size
        if start < 0:
            start = 0

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="テキストファイルをチャンクに分割してJSON出力します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python chunker.py -i input.txt -o output.json -c 512 -s 64
        """,
    )
    parser.add_argument("-i", "--input",        required=True,          help="入力テキストファイルパス")
    parser.add_argument("-o", "--output",        required=True,          help="出力JSONファイルパス")
    parser.add_argument("-c", "--chunk_size",    type=int, default=512,  help="チャンクサイズ（トークン数、デフォルト: 512）")
    parser.add_argument("-s", "--overlap_size",  type=int, default=64,   help="オーバーラップサイズ（トークン数、デフォルト: 64）")
    args = parser.parse_args()

    # バリデーション
    if not os.path.isfile(args.input):
        print(f"エラー: 入力ファイルが見つかりません: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.chunk_size <= 0:
        print("エラー: chunk_size は1以上の整数で指定してください", file=sys.stderr)
        sys.exit(1)

    if args.overlap_size < 0:
        print("エラー: overlap_size は0以上の整数で指定してください", file=sys.stderr)
        sys.exit(1)

    if args.overlap_size >= args.chunk_size:
        print("エラー: overlap_size は chunk_size より小さくしてください", file=sys.stderr)
        sys.exit(1)

    # 入力ファイル読み込み
    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    source_name = os.path.basename(args.input)

    # チャンク分割
    chunks = chunk_text(text, args.chunk_size, args.overlap_size)

    # JSON形式に変換
    results = []
    for idx, chunk in enumerate(chunks):
        results.append({
            "id":          idx,
            "text":        chunk,
            "source":      source_name,
            "token_count": count_tokens(chunk),
        })

    # JSON出力
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"完了: {len(results)} チャンクを {args.output} に出力しました", file=sys.stderr)


if __name__ == "__main__":
    main()
