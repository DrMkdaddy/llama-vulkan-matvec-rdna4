# Correctness outputs

`gen-old.txt` is from the stock `libggml-vulkan.so`; `gen-new.txt` is from the
patched library. Both are the concatenation of `reasoning_content` and
`content` for eight concurrent requests (`temperature: 0`, `seed: 7`,
256 output tokens each), one request per slot.

```bash
diff -u gen-old.txt gen-new.txt
# (empty)
```

The first version of this test set `max_tokens: 64` and compared only
`content`. The R1 distill was still inside its reasoning block at 64 tokens,
so both sides were empty and the comparison proved nothing. The files here are
from the corrected run with 256 tokens and both fields captured.
