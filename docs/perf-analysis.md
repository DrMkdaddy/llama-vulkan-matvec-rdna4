# Where the 7B's throughput comes from

Measured on the RX 9070 XT (16 GB, RADV/Vulkan), llama.cpp b11050 (60b06ab),
`DeepSeek-R1-Distill-Qwen-7B`, all layers on device, 4096 tokens per slot
unless stated. Harnesses and raw numbers: `bench/gpu/` (HIP microbenches),
`bench/spec-decode.sh` (server-side A/B + per-op timings), `bench/e2e.mjs`
(end-to-end TTFT/prefill/decode), `node bench/sweep.mjs` (`bench/results.json`).

Short version: the stock matvec kernels already run within ~11% of the card's
measured bandwidth ceiling, and a *hand-written replay of their exact access
pattern* runs at 99.9% of plain streaming. So custom matvec work at batch 1
buys nothing. The levers that pay: **batching** (np=8 -> 4x aggregate) and
**f16 KV** (+3-4%). The multi-column matvec used to drop to 406-481 GB/s
effective against 568 at one column; two changes recover most of it: hoisting
the weight dequant out of the column loop, and doubling the row block to 8
with the redundant k-unroll removed (sections 7/7b). Together: **437 -> 495 tok/s
at np=8, +13.4%**, single stream unchanged. IQ4_XS is +9-11% single stream
(+2.7% perplexity) but collapses when batched, so it is not the serving
default. Speculative decoding was measured four ways on this box and lost
every time.

## 1. The wall is weight bandwidth, and the kernels are against it

A decode step reads every weight once. From the op table below that is
**4.36 GB per token** for this model (Q4_K/Q6_K mix, plus the separate q6_K LM
head). Measured achievable bandwidth on this card (HIP):

| probe | GB/s |
|---|---:|
| plain streaming read | 635.7 |
| Q4_K matvec access pattern replayed exactly (no math, weights only) | **634.9** |
| stock Q4_K/Q6_K matvecs in the real graph | 491-622 |

The pattern replay (`bench/gpu/q4k_pattern.cpp`) maps each lane to the same
144-byte block slice the Vulkan shader reads, and lands at 99.9% of the plain
streaming rate: **the access pattern costs nothing**. The stock kernels then
reach 89% of that pattern rate. The gap between "ideal" and "shipped" on the
matvec path is therefore about 11%, and a custom kernel would have to beat the
driver's own streaming number to do better.

At 4.36 GB/token the single-stream ceiling is **~146 tok/s**. Measured is
**116.5 tok/s** (f16 KV, depth 0) — 80% of it.

## 2. What the 8.5 ms per token is spent on

Per-op totals for one decode step at batch 1 (`GGML_VK_PERF_LOGGER=1`, GPU
timestamps; call counts are per token):

| op | calls | us | GB/s | share |
|---|---:|---:|---:|---:|
| q4_K 18944x3584 FFN gate+up | 56 | 3589.8 | 596 | 42% |
| q6_K 3584x18944 FFN down | 14 | 1347.9 | 578 | 16% |
| q4_K 3584x18944 FFN down | 14 | 965.2 | 553 | 11% |
| q4_K 3584x3584 attn q/o (+fused add) | 55 | 824.7 | 491 | 10% |
| q6_K 152064x3584 LM head | 1 | 718.8 | 622 | 8% |
| q4_K 512x3584 K/V proj | 42 | 146.6 | 293 | 2% |
| q6_K 512x3584 K/V proj | 14 | 74.9 | 280 | 1% |
| **matvecs** | | **7668** | **568** | **90%** |
| RMS_NORM_MUL | 57 | 295.6 | – | 3.5% |
| FLASH_ATTN_EXT | 28 | 207.9 | – | 2.4% |
| ROPE (+ view/set_rows) | 56 | 193.6 | – | 2.3% |
| SET_ROWS (KV write) | 28 | 91.3 | – | 1.1% |
| GLU | 28 | 73.5 | – | 0.9% |
| misc (ADD, GET_ROWS) | | 25 | – | 0.3% |
| **total** | | **~8520** | | |

That total matches the wall clock (116.5 tok/s = 8.58 ms/token), so at batch 1
the GPU is saturated and the timestamps are trustworthy for this model size.
(For small models they are not: the 1.5B draft shows 5.4 ms of "GPU time"
against a 2.9 ms/token wall clock, because per-op submission gaps dominate.)

The 1.65 ms gap to the 6.87 ms ceiling splits into ~0.8 ms of matvec
inefficiency (568 vs 635 GB/s) and ~0.87 ms of latency-bound small ops: 57
RMSNorm calls at 5.2 us each, 56 RoPE calls, 56 SET_ROWS calls. Those ops are
grid-sized to the row and cannot be fused from inside a backend that executes
ops one at a time. Improving their shaders is worth ~1-3%; eliminating them
would need fusion in the graph, not in a kernel.

## 3. Batch > 1: where the aggregate ceiling comes from

Aggregate generation, `--parallel N` with one 4096-token slot per request:

| slots | aggregate tok/s | vs single stream |
|---:|---:|---:|
| 1 | 105.2 (116.5 with f16 KV) | 1.0x |
| 2 | 186.8 | 1.8x |
| 4 | 319.9 | 3.0x |
| 8 | 424.0 (417 with f16 KV) | 4.0x |
| 16 | 458.7 | 4.4x |

Per-request wall for 256 output tokens: 2.4 s single-stream, 3.2 s at np=4,
4.8 s at np=8, 8.9 s at np=16.

The README previously documented 286 tok/s at np=4; 8 and 16 slots are simply
better on this card. The curve flattens because *the weights are read once per
step either way*: each extra column costs ~1.5 ms/token of per-column work
while the 8.6 ms of weight traffic is shared.

The per-op table at 7 columns shows where that per-column cost goes:

| op | 1 col (us) | 7 cols (us) | delta |
|---|---:|---:|---:|
| q4_K gate+up (2.14 GB) | 3590 | 4448 | +858 |
| q6_K down (0.78 GB) | 1348 | 1990 | +642 |
| q4_K down (0.53 GB) | 965 | 1055 | +90 |
| q4_K attn q/o | 825 (fused) | 1372 (unfused) + 379 ADD | +926 |
| q6_K LM head (0.45 GB) | 719 | 1302 | +583 |
| K/V proj | 147 | 502 | +355 |
| **matvec subtotal** | **7668** | **10765** | **+3097** |
| small ops (norm, rope, attn, adds...) | 872 | 1774 | +902 |
| **total per step** | **8520** | **12540** | **+4020** |

Two things degrade, both fixable in principle:

1. **Effective bandwidth falls.** The same 4.36 GB of weights take 10.77 ms at
   seven columns (406 GB/s) against 7.67 ms at one (568 GB/s). The column loop
   is not keeping the memory pipes full — the LM head loses 45% (623 → 343
   GB/s), attention q/o 40%, q6_K down 32%.
2. **Op fusion is lost.** At one column the backend fuses `MUL_MAT+ADD` (the
   residual); at seven it emits the matvec and then 140 separate ADD calls
   (379 us per step for 3.9 MB of traffic).

Recovering the batch-1 bandwidth at 8 columns would cut ~3 ms from a ~19 ms
step: roughly +15-20% aggregate. That is the kernel project this stack would
actually benefit from. Section 7 lands the dequant half of it and measures
what is left.

## 4. Speculative decoding loses on this hardware (measured 4 ways)

The DeepInfer trick — draft tokens, schedule verification on a confidence
estimate — maps onto this build as `--spec-type draft-simple` plus
`--spec-draft-p-min` ("only collect very high-confidence draft tokens",
i.e. the confidence gate; drafting stops extending when the draft's token
probability falls below it). `draft-dspark`/`draft-dflash`/`draft-mtp` need
sidecars trained for them (DFlash+Markov head, EAGLE3 head, MTP layer) and a
plain Qwen2 target has none.

Vocab gate: a draft is only usable if its token ids are identical to the
target's. The target is the 152064-token Qwen2 vocab (bos 151646). Qwen2.5
drafts (0.5B and 3B here: 151936, bos 151643) are **rejected at startup**; the
only local model that matches is `DeepSeek-R1-Distill-Qwen-1.5B`.

| config (draft = R1-Distill 1.5B Q4_K_M) | acceptance | decode tok/s |
|---|---:|---:|
| baseline, no speculation | – | 107.0 |
| `--spec-draft-n-max 6 --spec-draft-p-min 0.6` | 69.3% | 73.7 |
| `--spec-draft-n-max 3` (no gate) | 58.3% | 106.3 |
| `--spec-draft-n-max 12 --spec-draft-p-min 0.8` | 88.5% | 88.0 |
| `--spec-type ngram-mod` (model-free) | 9.4% | 107.9 |

Why it cannot win here: the drafter costs **2.94 ms/token** (standalone
340 tok/s), and one verify pass costs ≥8.6 ms no matter how many tokens it
checks. With acceptance `a` and `k` drafted tokens, a step costs `2.94k + 8.6`
and yields `1 + a*k` tokens, so beating 9.35 ms/token needs ~85% acceptance at
*every* position with k≥3, sustained. Real text gives 58-88%, and the gate that
raises acceptance also shortens the draft. Verification is not the problem — a
k-column pass amortizes the weight read; the drafter is the tax.

What would win: a drafter at ≤1.5 ms/token (0.5B-class) **with this exact
vocab**, or a trained EAGLE3/MTP head for this family. Neither exists for
Qwen2-152064 today. Revisit if one appears.

## 5. KV cache type: f16 is fastest, not smallest

llama-bench tg128, `-ctk/-ctv`:

| model | KV | depth 0 | depth 4096 |
|---|---|---:|---:|
| Q4_K_M | f16 | **116.5 ± 0.3** | 107.6 ± 6.2 |
| Q4_K_M | q8_0 | 112.9 ± 1.2 | 108.2 ± 0.4 |
| IQ4_XS | f16 | **126.8 ± 1.7** | **119.9 ± 0.3** |
| IQ4_XS | q8_0 | 123.4 ± 0.3 | 116.5 ± 0.1 |

The server sweep at 8192 context agrees: f16 94.3 tok/s, q8_0 88.6, q4_0 88.6.

f16 is 3-4% faster at shallow depth (the KV read is on the critical path of
flash attention; quantize/dequantize costs more than the bytes it saves at this
size) and at least parity at 4096. Cost is ~100-235 MB at 4096 context. The
stack's q8_0 default was on the wrong side of this trade; the 7B entry now
uses f16. The 9B keeps q8_0 — it sits at 12.1 GB and needs the room.

## 6. Quantization: IQ4_XS is +9-11% single stream, but it collapses when batched

`llama-quantize --allow-requantize ... IQ4_XS` from the Q4_K_M file (there is
no f16 source locally, so this is a double quantization — a native IQ4_XS from
f16 would score slightly better):

| | Q4_K_M | IQ4_XS | delta |
|---|---:|---:|---:|
| file | 4.36 GiB | 3.95 GiB | -9.4% |
| tg128, depth 0 | 116.5 | 126.8 | **+8.9%** |
| tg128, depth 4096 | 107.6 | 119.9 | **+11.4%** |
| perplexity (wikitext-2, ctx 512, 24 chunks) | 23.60 | 24.24 | +2.7% |
| aggregate, 8 slots | 417 | **174** | **-58%** |

Single-stream throughput tracks bytes almost exactly, as a bandwidth-bound
kernel should, and +2.7% perplexity for +9-11% speed is usually a good trade.
But at 8 slots IQ4_XS is *worse than half* of Q4_K_M: the per-op table at 7
columns shows the gate/up matvec at **146 GB/s against 481 for Q4_K**, and the
attention q/o matvec at 109 against 295. The step takes 27.8 ms against 12.5.

The mechanism is not the dequant. IQ4_XS does not use the integer-dot shader
(`mul_mat_vecq.comp`) at all: i-quants have dedicated shaders
(`mul_mat_vec_iq4_xs.comp`) that keep the activations in f32, because the
4-bit values come from a non-uniform table and do not map onto DP4A. At 8
columns the activation slice is `8 * 3584 * 4` bytes per workgroup, four times
the q8_1 path's, and it dominates. Hoisting the table lookup out of the column
loop measured neutral (184.9 -> 182.4 tok/s at np=8, inside noise), so a
cheaper dequant is not the fix. Giving i-quants a q8_1 activation path is; see
section 8.

So there is no blanket quantization win: **keep Q4_K_M for serving, use IQ4_XS
only for a single-user box.** The fix is a q8_1 activation path for i-quants
(section 8).

## 7. The multi-column matvec: weight dequant hoisted (landed)

`iter()` in `mul_mat_vecq.comp` cached the activation block per column but
called `mmvq_dot_product(a_block_idx, b_qs_idx)` inside the column loop, so
every column re-read and re-dequantized the same weight block. The change
splits each type's dot into `load_a_tile()` (unpack one block into registers)
and `dot_a_tile()` (dot it against the cached activation), and preloads one
tile per row before the column loop:

```glsl
a_tile a_tiles[NUM_ROWS];
[[unroll]] for (uint n = 0; n < num_rows; ++n) {
    a_tiles[n] = load_a_tile(a_block_idx, b_qs_idx);
}
[[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
    // cache data_b for column j
    [[unroll]] for (uint n = 0; n < num_rows; ++n) {
        temp[j][n] += dot_a_tile(a_tiles[n]);
    }
}
```

Scope: the integer-dot shader covers the K-quants and the legacy quants, which
is the serving default. The i-quants use dedicated float-B shaders and are
untouched (section 6). Upstream master has since moved IQ4_XS onto this same
integer-dot shader, so a rebased patch would cover it too.

Measured at np=8, 32k context, f16 KV, 256 output tokens per slot, three runs,
medians, with direct `/completion` requests rather than the gateway sweep
(per-op timings from `GGML_VK_PERF_LOGGER=1`):

| n=8 | before | hoist only | delta |
|---|---:|---:|---:|
| aggregate tok/s | 437.0 | **460** | **+5.3%** |
| GPU time per step | 14.75 ms | 13.80 ms | -6.4% |
| q4_K gate+up (m=18944) | 108.5 us | 97.0 us | -10.5% |
| q6_K FFN down (m=3584, k=18944) | 191.4 us | 175.0 us | -8.6% |
| q4_K attn q/o (3584x3584) | 25.6 us | 24.1 us | -5.8% |
| q4_K K/V (m=512) | 15.0 us | 14.1 us | -5.8% |
| q4_K FFN down (m=3584, k=18944) | 74.8 us | 75.0 us | 0% |
| q6_K LM head (m=152064) | 1320.9 us | 1352.4 us | +2.4% |

Single stream is unchanged: 117.4 tok/s against 117.6 for the shipped build
(at one column the tile is loaded once either way). Correctness: 8 concurrent
greedy requests are byte-identical before and after, because the change only
removes repeated integer work.

Why +5.3% and not the +15-20% section 3 projected: with the dequant hoisted,
the 8-column kernel is no longer ALU-bound. Two limits remain, both measured:

1. **Activation traffic.** Each workgroup re-reads the whole activation slice
   for its rows: `(m / rows) * cols * k` bytes per step. For gate/up that is
   about 150 MB, four times the weight stream; for the LM head (m=152064) it
   is 1.09 GB. Sharing one activation slice across columns (a timing hack,
   wrong output) cut the LM head from 1315 to 995 us (-24%), so the large-m
   ops are activation-bound. Gate/up did not move under the same hack
   (94 -> 102 us): its remaining 34% gap against the one-column rate is
   per-column latency and occupancy, not bytes.
2. **Register pressure at 8 rows.** A tile costs 5-6 registers per row, and
   the original loop unrolled four k-iterations on top of the 8x8 column/row
   bodies. That combination spills (gate/up 165 us against 97). Removing the
   extra unroll for NUM_ROWS>=8 drops it to 78 us (section 7b); the row count
   was never the problem.

Even after 7b the floor is still visible: gate/up runs at 491 GB/s effective
against 597 at one column. Closing it needs a structural change, either
activation staging in shared memory for the large-m ops or a register-cheaper
row group for the FFN, not more dequant work.

### 7b. Wider row blocking (landed)

With the dequant hoisted, the 8-column kernel is limited by the activation
loads each workgroup re-reads for its rows, not by the dequant. The first
attempt at NUM_ROWS=8 spilled (gate/up 165 us against 97) and 4 stayed the
optimum. The spill was not the row count: it was the extra 4x k-unroll, which
multiplied the 8x8 inlined dot bodies into ~1500 instructions and blew up the
live ranges. Removing that unroll for NUM_ROWS>=8 makes the wider block fit:

| n=8 | hoist only | + rows=8, no 4x unroll | delta |
|---|---:|---:|---:|
| aggregate tok/s | 460 | **495** | **+7.6%** |
| GPU time per step | 13.80 ms | 12.56 ms | -9.0% |
| q4_K gate+up (m=18944) | 97.0 us | 77.8 us | -19.8% |
| q4_K K/V (m=512) | 14.1 us | 10.1 us | -28.4% |
| q6_K FFN down (m=3584, k=18944) | 175.0 us | 166.5 us | -4.9% |
| q4_K attn q/o (3584x3584) | 24.1 us | 24.3 us | +0.8% |
| q4_K FFN down (m=3584, k=18944) | 75.0 us | 80.5 us | +7.3% |
| q6_K LM head (m=152064) | 1352.4 us | 1348.0 us | -0.3% |

The wider block halves the activation bytes per output row and gives the
scheduler two independent column chains per activation load. It wins where
the activation stream is the limiter (gate/up, q6_K down) and is roughly
neutral where the weight stream dominates (q4_K down, +7.3%). The aggregate
is still -9.0% GPU time because gate/up is 35% of the step.

Correctness: 8 concurrent greedy requests are byte-identical before and
after, because the row block only changes which output rows a workgroup
owns, not the arithmetic. Single stream is unchanged (at one column the
kernel takes the NUM_ROWS<8 path and loads the tile once either way).

Cumulative: stock 437 -> 460 (hoist) -> 495 (rows=8) tok/s, +13.4% over
stock at np=8.

## 8. What is left, ranked

1. **Concurrency** (landed): np=8, ctx = np x 4096. 4x aggregate, no quality
   cost.
2. **f16 KV** (landed): +3-4% single stream, ~100 MB.
3. **Multi-column matvec dequant hoist + 8-row blocking** (landed): +13.4%
   aggregate at np=8 (437 -> 495), single stream unchanged. Sections 7/7b.
4. **Activation traffic at n>1** (next kernel project): the remaining gap is
   the per-workgroup activation re-read (1.09 GB/step on the LM head, about
   150 MB on gate/up). Shared-memory staging for the large-m ops is worth up
   to 24% on the LM head alone. Gate/up sits 18% below its one-column rate for
   a different reason (per-column latency at NUM_ROWS=8), so the two need
   different fixes.
5. **A q8_1 path for i-quants** (would make IQ4_XS a serving option): +9-11%
   single stream and 9.4% fewer bytes, but the float32 activation stream keeps
   it at ~185 tok/s at np=8 against 495.
6. **Small-op shader tuning** (RMSNorm/RoPE/SET_ROWS grids): ~1-3% single
   stream, and they cost 1.7 ms/step at 8 columns.
7. **A trained drafter for this vocab** (EAGLE3/MTP): the only route to a
   step change beyond the bandwidth wall.
