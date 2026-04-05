#!/bin/bash
# 埋め込みモデル（paraphrase-multilingual-MiniLM-L12-v2）をローカルにダウンロードする


MODEL_DIR="./models/embedded/paraphrase-multilingual-MiniLM-L12-v2"

mkdir -p "$(dirname "$MODEL_DIR")"

if [ -d "$MODEL_DIR" ]; then
    echo "既にダウンロード済みです: $MODEL_DIR"
    exit 0
fi

echo "埋め込みモデルをダウンロードします..."

python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_dir="$MODEL_DIR",
)
EOF

echo "ダウンロード完了: $MODEL_DIR"
