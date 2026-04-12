#pragma once

// ============================================================================
// CUDA Utility Macros and Common Kernels
// ============================================================================
// Shared across all GPU versions (V1-V6).
//
// Why macros for error checking? CUDA functions return error codes silently.
// Without checking, a failed cudaMalloc or kernel launch produces garbage
// results with no error message — incredibly hard to debug. These macros
// convert silent failures into loud, informative crashes with file/line info.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error at %s:%d — %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while(0)

#define CUBLAS_CHECK(call) do {                                             \
    cublasStatus_t status = (call);                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
        fprintf(stderr, "cuBLAS Error at %s:%d — status %d\n",             \
                __FILE__, __LINE__, static_cast<int>(status));              \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while(0)

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

// Ceiling division: div_ceil(7, 3) = 3 (used for computing grid dimensions)
inline __host__ __device__ int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

// Standard thread block size for simple element-wise kernels.
// 256 is a good default: it gives reasonable occupancy on most GPUs
// while being a multiple of the warp size (32).
constexpr int BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// MSE Loss Reduction Kernel
// ---------------------------------------------------------------------------
// Computes sum of squared differences: sum((output[i] - target[i])^2)
// Uses tree reduction in shared memory — the standard parallel reduction
// pattern on GPUs:
//
//   Step 1: Each thread computes one (output-target)^2 → shared memory
//   Step 2: Tree reduction halves active threads each iteration
//   Step 3: Block result atomicAdd'd to global accumulator
//
// Why atomicAdd? Each block produces one partial sum. We need to combine
// all block results. atomicAdd is simple and correct (though not the fastest
// for very large arrays — fine for our loss computation).
// ---------------------------------------------------------------------------
__global__ void mse_loss_kernel(const float* output, const float* target,
                                float* loss, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load: each thread computes one squared difference
    sdata[tid] = 0.0f;
    if (idx < N) {
        float diff = output[idx] - target[idx];
        sdata[tid] = diff * diff;
    }
    __syncthreads();

    // Tree reduction in shared memory
    // At each step, the first s threads add elements s positions away.
    // After log2(blockDim) steps, sdata[0] contains the block's total.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes its partial sum to the global result
    if (tid == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

// ---------------------------------------------------------------------------
// Host-side wrapper for MSE loss computation
// ---------------------------------------------------------------------------
// Returns: (1 / 2N) * sum((output - target)^2)
// The 1/2 factor makes the derivative cleaner: dL/dy = (y - t) / N
// ---------------------------------------------------------------------------
inline float compute_mse_loss(const float* d_output, const float* d_target,
                              int N, float* d_loss) {
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    int blocks = div_ceil(N, BLOCK_SIZE);
    mse_loss_kernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_output, d_target, d_loss, N);
    float loss;
    CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return loss / (2.0f * N);
}
