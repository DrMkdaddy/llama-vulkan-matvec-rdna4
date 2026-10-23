# Patch

`0001-vulkan-mmvq-multicolumn.patch` applies to llama.cpp b11050
(`60b06ab`) with `git apply -p1`.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout --detach 60b06ab
git apply /path/to/0001-vulkan-mmvq-multicolumn.patch
```

Files changed:

| File | What |
|---|---|
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | selects 8 output rows for Q4_K/Q5_K in the 4–8-column shader variants |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp` | preloads a weight tile per output row; skips the 4× k-unroll at 8 rows |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl` | splits each quant's dot into `load_a_tile()` + `dot_a_tile()` |

The patch is against a moving target. If a newer llama.cpp has changed
`mul_mat_vecq`, rebase the three files rather than forcing the diff; the
change is local to the shader and the pipeline row-count selection.

The patch is not upstreamed. It is published under the MIT license, the same
as llama.cpp.
