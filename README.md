# llama.cpp Vulkan matvec: dequant hoisting + 8-row blocking

A patch to llama.cpp's Vulkan `mul_mat_vecq` shader. It removes redundant
K-quant dequantization and activation reads at multiple slots.

Measured on one AMD RX 9070 XT (RDNA4, gfx1201) serving
DeepSeek-R1-Distill-Qwen-7B Q4_K_M to eight concurrent slots at 32k context
with an f16 KV cache:

| np=8 | Aggregate tok/s | GPU time per step |
|---|---:|---:|
| stock llama.cpp b11050 | 437.0 | 14.75 ms |
| + dequant hoist | 460 | 13.80 ms |
| + 8-row blocking | **495** | **12.56 ms** |

**+13.4% aggregate over stock.** Single-stream throughput is unchanged
(117.4 vs 117.6 tok/s), and eight concurrent greedy requests produce
byte-identical output before and after.

Upstream PR (Phase 1 dequant hoist):
[ggml-org/llama.cpp#31022](https://github.com/ggml-org/llama.cpp/pull/31022)

## Why it helps

At batch 1, decode is a weight-bandwidth problem: every step reads every weight
once. The stock Vulkan kernels already run within about 11% of this card's
measured streaming ceiling, and a HIP replay of the exact Q4_K access pattern
reaches 634.9 GB/s against 635.7 GB/s for plain streaming. The access pattern is
not the constraint.

At np=8, two redundancies appear in `mul_mat_vecq`:

1. **Dequantization sat inside the token-column loop.** Each of the eight token
   columns re-read and re-unpacked the same K-quant block.
2. **Activation reads repeated per output row.** A workgroup owned four output
   rows and re-read the whole activation slice for them.

## What changed

Three files, +192/−104 lines, against llama.cpp b11050 (`60b06ab`).

- `mul_mat_vecq_funcs.glsl`: split `mmvq_dot_product()` into `load_a_tile()`
  and `dot_a_tile()`. The tile carries the unpacked `qs` words and scale/dm
  values.
- `mul_mat_vecq.comp`: preload one `a_tile` per output row and reuse it for
  every column. Skip the 4x k-unroll when `NUM_ROWS >= 8`; the extra unroll
  spills at eight rows.
- `ggml-vulkan.cpp`: select eight rows for Q4_K and Q5_K in the 4-8-column
  variants. The guard reuses the existing RDNA3-class check, which includes
  gfx1201 on this build.

Q6_K gets the hoist but keeps its existing row block. i-quants use separate
shaders and are untouched.

The first change is smaller and simpler. The second adds more throughput,
raising aggregate to 495 tok/s, and exposes the next limit: the LM head is
activation-bound and needs shared-memory staging rather than a wider row block.

## Correctness

- Eight concurrent greedy requests (`temperature: 0`, `seed: 7`, 256 tokens)
  produced byte-identical output before and after. Outputs are in
  [`results/correctness/`](results/correctness/).
- The patch changes which workgroup owns which output rows and when a weight
  tile is unpacked, not the per-element arithmetic order.
- PPL on wikitext-2 (ctx 512, 24 chunks) is 23.6013 on the patched build,
  matching 23.60 on the stock Q4_K_M stack.

## Reproduce

```bash
# Clone, patch, and build Vulkan.
./reproduce.sh

# Serve the 7B with eight slots.
build/bin/llama-server \
  -m "$HOME/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf" \
  --alias deepseek-r1-7b --host 127.0.0.1 --port 8082 \
  -c 32768 --parallel 8 -ngl 99 -fa on \
  -ctk f16 -ctv f16 --cache-reuse 256 --reasoning off

# Measure aggregate throughput.
CONCURRENT=8 RUNS=3 MAX_TOKENS=256 CTX=32768 PARALLEL=8 \
KVTYPE_K=f16 KVTYPE_V=f16 ./bench/spec-decode.sh np8
```

## Limits

- One GPU, one stack, one model: RX 9070 XT, RADV/Vulkan, 7B Q4_K_M, np=8.
  This is a single-card result; do not assume the same percentage elsewhere.
- Single-stream throughput is unchanged by design. The patch targets concurrent
  serving.
- Not upstreamed. This is a patch against b11050 (`60b06ab`), not a fork.
  Newer builds will need a rebase.
- The wide-row path reuses the `AMD_RDNA3` classification, which includes
  gfx1201 on this build. Verify the mapping before porting to another
  architecture.
- Next limit: activation traffic. The LM head re-reads about 1.09 GB of
  activations per step; sharing one activation slice across columns in shared
  memory measured up to 24% faster on that op alone.

Per-op tables, methodology, and negative results (IQ4_XS, speculative decoding,
activation staging) are in [`docs/perf-analysis.md`](docs/perf-analysis.md).

## License

MIT. The patch modifies llama.cpp, which is MIT-licensed.
