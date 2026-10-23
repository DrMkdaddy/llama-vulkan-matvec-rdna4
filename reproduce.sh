#!/usr/bin/env bash
# Clone llama.cpp at the tested commit, apply the patch, and build the Vulkan backend.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$ROOT/work}"
SRC="$WORK/llama.cpp"
COMMIT="${LLAMA_COMMIT:-60b06ab}"

mkdir -p "$WORK"

if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi

if [ -n "$(git -C "$SRC" status --porcelain)" ]; then
  echo "error: $SRC has local changes; use a fresh WORK directory" >&2
  exit 1
fi

git -C "$SRC" fetch --tags origin
git -C "$SRC" checkout --detach "$COMMIT"

PATCH="$ROOT/patches/0001-vulkan-mmvq-multicolumn.patch"
git -C "$SRC" apply "$PATCH"

cmake -S "$SRC" -B "$SRC/build" \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF

cmake --build "$SRC/build" --config Release -j"$(nproc)" \
  --target llama-server llama-bench

# Stable path for the commands below and in the README.
ln -sfn "$SRC/build" "$ROOT/build"

cat <<EOF

Built:
  $ROOT/build/bin/llama-server
  $ROOT/build/bin/llama-bench

Serve:
  $SRC/build/bin/llama-server \\
    -m "\$HOME/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf" \\
    --alias deepseek-r1-7b --host 127.0.0.1 --port 8082 \\
    -c 32768 --parallel 8 -ngl 99 -fa on \\
    -ctk f16 -ctv f16 --cache-reuse 256 --reasoning off

Measure:
  CONCURRENT=8 RUNS=3 MAX_TOKENS=256 CTX=32768 PARALLEL=8 \\
  KVTYPE_K=f16 KVTYPE_V=f16 \\
    $ROOT/bench/spec-decode.sh np8
EOF
