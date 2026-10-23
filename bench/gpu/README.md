# GPU microbenches (HIP)

Two standalone HIP programs that answer "how much headroom is there in the
matvec kernels?" on gfx1201 (RX 9070 XT). Build and run on the host; they do
not touch llama.cpp.

```bash
hipcc -O3 --offload-arch=gfx1201 bwtest.cpp -o bwtest
./bwtest 1024 5          # buffer size in MiB, iterations

hipcc -O3 --offload-arch=gfx1201 q4k_pattern.cpp -o q4k_pattern
./q4k_pattern 4608       # MiB of Q4_K blocks to stream
```

`bwtest` measures plain streaming read and read+write bandwidth. That is the
ceiling a batch-1 decode step races against, because the step reads every
weight once.

`q4k_pattern` replays the exact lane-to-address mapping of the Vulkan Q4_K
matvec shader without doing any dequant math: 16 threads per 144-byte block,
2 rows per workgroup, each thread taking 8 bytes of `qs` plus a header byte.
If the pattern matches plain streaming, the access pattern is not a constraint.

Measured on this card (llama.cpp b11050, RADV):

```
bwtest       read-only 616.5 GB/s, read+write 573.2 GB/s
q4k_pattern  stream 635.7 GB/s, Q4_K pattern 634.9 GB/s
```

The 0.1% difference is the finding: the stock Vulkan matvec kernels reach
89% of that pattern rate, so batch-1 matvec work is done. See
[`docs/perf-analysis.md`](../../docs/perf-analysis.md).
