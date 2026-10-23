# Benchmarks

## End to end

`spec-decode.sh` starts `llama-server`, waits for health, and runs
`spec-client.mjs`. It uses `/completion` and reports decode throughput plus
draft acceptance when speculative decoding is enabled.

```bash
CONCURRENT=8 RUNS=3 MAX_TOKENS=256 CTX=32768 PARALLEL=8 \
KVTYPE_K=f16 KVTYPE_V=f16 \
  ./spec-decode.sh np8
```

Environment variables: `LLAMA_SERVER`, `MODEL`, `PORT`, `PROMPT_KIND`
(`code` or `json`), `RUNS`, `MAX_TOKENS`, `CONCURRENT`, `CTX`, `PARALLEL`,
`KVTYPE_K`, `KVTYPE_V`.

`e2e.mjs` measures prompt processing and decode against a running server using
the server's own timings. Set `E2E_URL`, `E2E_MODEL`, `E2E_N`, `E2E_MAX`.

`sweep.mjs` runs the broader stack sweep (generation, prefill as context
grows, concurrency, KV cache types, prefix reuse) and writes
`bench/results.json`. Set `LLAMA_SERVER`, `MODEL_DIR`, `SWEEP_MODEL`,
`SWEEP_ALIAS`, `SWEEP_PORT`. `results.json` in this directory is the baseline
sweep from the stock build; it is context, not the A/B in the README.

## GPU microbenches

`gpu/bwtest.cpp` measures plain read and read+write bandwidth.
`gpu/q4k_pattern.cpp` replays the exact lane-to-address mapping of the Vulkan
Q4_K matvec without doing the math. If the replay matches plain streaming, the
access pattern is not the constraint.

```bash
hipcc -O3 --offload-arch=gfx1201 gpu/bwtest.cpp -o bwtest
hipcc -O3 --offload-arch=gfx1201 gpu/q4k_pattern.cpp -o q4k_pattern
./bwtest 1024 5
./q4k_pattern 4608
```
