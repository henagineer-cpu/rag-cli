#!/bin/bash
# setup.sh — conda による Python 3.14 環境のセットアップスクリプト
#
# 使い方:
#   bash setup.sh

set -e

ENV_NAME="rag-env"
MINICONDA_URL_LINUX="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL_MAC="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"

# ── Step 1: conda の確認 / Miniconda インストール ──────────────────────────
if ! command -v conda &>/dev/null; then
    echo "conda が見つかりません。Miniconda をインストールします..."

    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        curl -fsSL "$MINICONDA_URL_LINUX" -o "$MINICONDA_INSTALLER"
    elif [ "$OS" = "Darwin" ]; then
        curl -fsSL "$MINICONDA_URL_MAC" -o "$MINICONDA_INSTALLER"
    else
        echo "エラー: 未対応のOS ($OS)。手動で Miniconda をインストールしてください。"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    rm -f "$MINICONDA_INSTALLER"

    # 現在のシェルセッションで conda を有効化
    export PATH="$HOME/miniconda3/bin:$PATH"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"

    echo "Miniconda のインストール完了: $HOME/miniconda3"
else
    eval "$(conda shell.bash hook)"
fi

echo "conda バージョン: $(conda --version)"

# ── Step 2: 環境の作成 ────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "既存の conda 環境 '$ENV_NAME' を更新します..."
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "conda 環境 '$ENV_NAME' を作成します..."
    conda env create -f environment.yml
fi

# ── Step 3: 完了メッセージ ────────────────────────────────────────────────
echo ""
echo "セットアップ完了。以下で環境を有効化してください:"
echo "  source ~/.bashrc"
echo "  conda activate $ENV_NAME"
echo ""
