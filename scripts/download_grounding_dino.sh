#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models/grounding-dino-base"
BASE_URL="https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main"

mkdir -p "$MODEL_DIR"

files=(
  "config.json"
  "model.safetensors"
  "preprocessor_config.json"
  "special_tokens_map.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.txt"
)

for file in "${files[@]}"; do
  echo "[download] $file"
  curl -L "$BASE_URL/$file" -o "$MODEL_DIR/$file"
done

echo "[done] Model files saved to $MODEL_DIR"
