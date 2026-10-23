# Methodology

## Hardware and software

| | |
|---|---|
| GPU | AMD RX 9070 XT, 16 GB (gfx1201, RDNA4) |
| Driver | RADV / Vulkan |
| llama.cpp | b11050 (`60b06ab`), Release build, `GGML_VULKAN=ON` |
| Model | DeepSeek-R1-Distill-Qwen-7B Q4_K_M, 4.36 GiB, all layers on GPU |
| Runtime | 32k context, 8 slots (4k per slot), f16 KV, flash attention on, `--cache-reuse 256`, `--reasoning off` |

## A/B setup

The stock and patched builds share the same `llama-server` binary and differ
only in `libggml-vulkan.so`. The stock library was built before the patch and
kept in a separate directory; the patched library is the one in `build/bin`.
The harness switches between them with `LD_LIBRARY_PATH`, so the model,
tokenizer, sampler, and server code are identical.

The three variants in the results table are cumulative:

1. **stock** — b11050 as shipped.
2. **+ hoist** — `load_a_tile()` / `dot_a_tile()` refactor only.
3. **+ 8-row** — the hoist plus the wider Q4_K/Q5_K row block and the
   `NUM_ROWS >= 8` unroll change.

## Throughput

Each run sends eight concurrent `/completion` requests, 256 output tokens
each, and records the aggregate tokens divided by wall time. Three runs are
taken and the median is reported. The per-op tables come from a separate
steady-state batch with `GGML_VK_PERF_LOGGER=1`; the last `n=8` table in the
server log is the one used.

The stock figure in this repo (437 tok/s) and the figure in the broader stack
write-up (424 tok/s) come from different harnesses: this one uses direct
`/completion` requests with a fixed 256-token output, while the stack sweep
uses its own client and timing. Compare numbers within one table, not across
tables.

## Correctness

Eight concurrent requests with `temperature: 0`, `seed: 7`, and 256 output
tokens. The R1 distill emits its reasoning trace in `reasoning_content`; the
test concatenates `reasoning_content + content` before comparing, because the
first version of the test compared only `content` while the model was still
inside its think block and therefore compared empty strings.

The final outputs are in [`../results/correctness/`](../results/correctness/).
`diff -u gen-old.txt gen-new.txt` is empty.

PPL is measured on wikitext-2 test, context 512, 24 chunks, using the patched
build: 23.6013. The stock Q4_K_M figure recorded for the same model is 23.60.

## What is not measured

- Other GPUs, other drivers, other quant types.
- Batch sizes other than 1 and 8.
- Speculative decoding, which was measured separately on this card and lost
  (see `docs/perf-analysis.md`).
- Long-context behavior beyond 32k.
