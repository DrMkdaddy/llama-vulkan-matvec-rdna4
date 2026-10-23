# Results index

| File | What |
|---|---|
| `perop-np8-final.txt` | final patched build, steady-state `n=8` per-op table from `GGML_VK_PERF_LOGGER=1`; total 12.56 ms |
| `perop-n1.log` | single-stream per-op table, showing the `n=1` path the patch does not change |
| `ppl-np8.log` | wikitext-2 PPL (ctx 512, 24 chunks) and the np=8 aggregate runs |
| `correctness/gen-old.txt` | eight concurrent greedy outputs, stock `libggml-vulkan.so` |
| `correctness/gen-new.txt` | the same eight requests, patched library |

The aggregate numbers in the README (437 → 460 → 495 tok/s) come from the
three-run medians described in `docs/methodology.md`. The per-op table here is
the final variant; the intermediate hoist-only table is in
`docs/perf-analysis.md`.
