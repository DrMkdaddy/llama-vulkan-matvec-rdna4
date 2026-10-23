// Streaming bandwidth microbench for gfx1201 (RX 9070 XT).
// Answers: what read/write bandwidth can this card actually sustain?
// A Q4_K matvec at batch 1 is a pure streaming-read problem, so this is the
// ceiling any custom kernel is racing against.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void read_only(const float4* __restrict__ in, float* __restrict__ out, size_t n4) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[blockIdx.x] = acc;  // never true; stops DCE
}

__global__ void read_write(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = in[i];
        v.x += 1.f; v.y += 1.f; v.z += 1.f; v.w += 1.f;
        out[i] = v;
    }
}

int main(int argc, char** argv) {
    size_t bytes = (argc > 1 ? atol(argv[1]) : 1024) * 1024ull * 1024ull;  // buffer size
    size_t n4 = bytes / 16;
    int iters = argc > 2 ? atoi(argv[2]) : 5;

    float4 *a, *b;
    float* sink;
    HIP_CHECK(hipMalloc(&a, bytes));
    HIP_CHECK(hipMalloc(&b, bytes));
    HIP_CHECK(hipMalloc(&sink, 4096));
    HIP_CHECK(hipMemset(a, 1, bytes));

    int dev; HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("device: %s  (CUs=%d, mem=%.1f GB)\n", prop.name, prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

    int threads = 256;
    int blocks = prop.multiProcessorCount * 16;

    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));

    float best_read = 0, best_rw = 0;
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipEventRecord(t0));
        read_only<<<blocks, threads>>>(a, sink, n4);
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipEventSynchronize(t1));
        float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        float gbs = bytes / (ms * 1e-3f) / 1e9f;
        if (gbs > best_read) best_read = gbs;

        HIP_CHECK(hipEventRecord(t0));
        read_write<<<blocks, threads>>>(a, b, n4);
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipEventSynchronize(t1));
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        gbs = 2 * bytes / (ms * 1e-3f) / 1e9f;
        if (gbs > best_rw) best_rw = gbs;
    }

    printf("read-only : %7.1f GB/s\n", best_read);
    printf("read+write: %7.1f GB/s  (counts both directions)\n", best_rw);
    printf("\n7B Q4_K_M = 4.51 GB of weights per token:\n");
    printf("  ceiling at read-only BW: %6.1f tok/s\n", best_read / 4.51);
    return 0;
}
