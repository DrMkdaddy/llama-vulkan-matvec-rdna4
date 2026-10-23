#!/usr/bin/env bash
# Speculative-decoding tuning harness.
#
#   bench/spec-decode.sh <label> [llama-server args...]
#
# Starts llama-server with the given extra args, waits for health, runs the
# spec client, prints the throughput + draft acceptance, and shuts the server down.
# Baseline:   bench/spec-decode.sh baseline
# With draft: bench/spec-decode.sh draft -md $MODEL_DIR/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
#               --spec-type draft-simple --spec-draft-n-max 6 --spec-draft-p-min 0.6
# Model-free: bench/spec-decode.sh ngram --spec-type ngram-mod
set -euo pipefail

LABEL="${1:?usage: spec-decode.sh <label> [llama-server args...]}"
shift

BIN="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf}"
PORT="${PORT:-8090}"
PROMPT_KIND="${PROMPT_KIND:-code}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CONCURRENT="${CONCURRENT:-1}"
CTX="${CTX:-4096}"
PARALLEL="${PARALLEL:-1}"

CLIENT="$(dirname "$0")/spec-client.mjs"

"$BIN" \
  -m "$MODEL" -a deepseek-r1-7b \
  --port "$PORT" \
  -c "$CTX" --parallel "$PARALLEL" \
  -ngl 99 -fa on -ctk "${KVTYPE_K:-q8_0}" -ctv "${KVTYPE_V:-q8_0}" \
  --reasoning off \
  "$@" > "/tmp/spec-$LABEL.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  sleep 0.5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "server failed to start; log:"; tail -20 "/tmp/spec-$LABEL.log"; exit 1; }

echo "### $LABEL"
node "$CLIENT" --port="$PORT" --runs="$RUNS" --max-tokens="$MAX_TOKENS" --prompt="$PROMPT_KIND" --concurrent="$CONCURRENT"
