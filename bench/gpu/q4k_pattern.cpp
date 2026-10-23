// Does the Q4_K matvec *access pattern* itself cap bandwidth?
//
// The Vulkan Q4_K matvec kernel reads a 144-byte block as: 16 threads split the
// 128 qs bytes (8 bytes each), plus a 16-byte header (2 halves + 12 scale bytes).
// Each workgroup covers 2 rows. This replays that pattern without any dequant
// math: if this hits the same GB/s as a plain streaming read, the pattern is
// not the problem and no custom kernel can beat the stock one by much.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

#define BLOCK_BYTES 144
#define QK 256

// Plain streaming read of the same total bytes.
__global__ void stream_read(const float4* __restrict__ in, float* __restrict__ out, unsigned n4) {
    float acc = 0.f;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;
}

// Q4_K matvec read pattern: same lane->address mapping the Vulkan kernel uses.
// One "row" = n_blocks blocks of 144 B. A workgroup covers NUM_ROWS rows.
// 16 threads cooperate on one block. Even lane il == 0..3 maps to
// (v_im, v_in) = (il/2, il%2) and l0 = 4*(2*(itid%4) + v_in), so the thread's
// 8-byte qs chunk sits at byte offset 16 + v_im*64 + l0, and it also touches
// one header byte (16 threads x 9 bytes = the whole 144-byte block).
__global__ void q4k_pattern(const uchar* __restrict__ in, float* __restrict__ out,
                            unsigned n_blocks_per_row, unsigned n_rows) {
    const unsigned NUM_ROWS = 2;
    const unsigned wg = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned first_row = NUM_ROWS * wg;

    const unsigned it_size = blockDim.x / 16;   // 128/16 = 8
    const unsigned itid = tid % 16;
    const unsigned ix   = tid / 16;             // which block slot
    const unsigned il   = itid / 4;             // 0..3
    const unsigned ir   = itid - 4 * il;        // 0..3
    const unsigned v_im = il / 2;               // which 64-byte half
    const unsigned v_in = il % 2;
    const unsigned l0   = 4 * (2 * ir + v_in);  // 0,4,8,12
    const unsigned qs_off = 16 + v_im * 64 + l0;

    float acc = 0.f;
    for (unsigned n = 0; n < NUM_ROWS; ++n) {
        const unsigned row = first_row + n;
        if (row >= n_rows) break;
        const uchar* row_ptr = in + (size_t)row * n_blocks_per_row * BLOCK_BYTES;
        for (unsigned b = ix; b < n_blocks_per_row; b += it_size) {
            const uchar* blk = row_ptr + (size_t)b * BLOCK_BYTES;
            uint2 qs = *(const uint2*)(blk + qs_off);
            acc += (float)(qs.x + qs.y + blk[itid]);   // header byte keeps the block fully covered
        }
    }
    if (acc == 12345.678f) out[0] = acc;
}

int main(int argc, char** argv) {
    // 4.5 GB of "weights", same as the 7B Q4_K_M
    size_t bytes = (argc > 1 ? atol(argv[1]) : 4608ull) * 1024ull * 1024ull;
    unsigned n_blocks = bytes / BLOCK_BYTES;
    // pick a realistic row structure: 18944 rows x 74 blocks (ffn gate/up, k=18944)
    unsigned n_blocks_per_row = 74;
    unsigned n_rows = n_blocks / n_blocks_per_row;
    size_t alloc = (size_t)n_rows * n_blocks_per_row * BLOCK_BYTES;

    void *a; float *sink;
    HIP_CHECK(hipMalloc(&a, alloc));
    HIP_CHECK(hipMalloc(&sink, 4096));
    HIP_CHECK(hipMemset(a, 1, alloc));

    int dev; HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    const int threads = 128;                  // same as the Vulkan matvec WG
    const unsigned wgs = (n_rows + 1) / 2;    // one WG per 2 rows
    const unsigned stream_blocks = prop.multiProcessorCount * 16;

    hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));

    double best_stream = 0, best_pattern = 0;
    for (int i = 0; i < 5; i++) {
        float ms;
        HIP_CHECK(hipEventRecord(t0));
        stream_read<<<stream_blocks, 256>>>((const float4*)a, sink, alloc / 16);
        HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        best_stream = fmax(best_stream, alloc / (ms * 1e-3) / 1e9);

        HIP_CHECK(hipEventRecord(t0));
        q4k_pattern<<<wgs, threads>>>((const uchar*)a, sink, n_blocks_per_row, n_rows);
        HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        best_pattern = fmax(best_pattern, alloc / (ms * 1e-3) / 1e9);
    }

    printf("bytes streamed per pass: %.2f GB over %u rows x %u blocks\n", alloc / 1e9, n_rows, n_blocks_per_row);
    printf("plain streaming read : %7.1f GB/s\n", best_stream);
    printf("Q4_K matvec pattern  : %7.1f GB/s\n", best_pattern);
    printf("\nsame math in tok/s for a 7B Q4_K_M (4.51 GB/token):\n");
    printf("  streaming ceiling : %6.1f tok/s\n", best_stream / 4.51);
    printf("  pattern ceiling   : %6.1f tok/s\n", best_pattern / 4.51);
    return 0;
}
