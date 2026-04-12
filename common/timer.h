#pragma once

// ============================================================================
// Timing Utilities
// ============================================================================
// CPUTimer: Uses std::chrono for host-side wall-clock timing.
//   - Available in all versions (CPU and GPU).
//   - Measures end-to-end time including kernel launches + synchronization.
//
// CUDATimer: Uses cudaEvent_t for GPU-side timing.
//   - Only available when compiled with nvcc (__CUDACC__ defined).
//   - Measures actual GPU execution time, excluding CPU overhead.
//   - More accurate for kernel-level benchmarking because it uses the
//     GPU's internal clock, not the host's.
//
// Why both? CPU timer captures the full picture (including launch overhead,
// memory transfers, etc.), while CUDA timer isolates kernel performance.
// The difference between them reveals CPU-side bottlenecks.
// ============================================================================

#include <chrono>

class CPUTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ---------------------------------------------------------------------------
// CUDA Event Timer — only available when compiled with nvcc
// ---------------------------------------------------------------------------
// cudaEvent_t records timestamps on the GPU's timeline, so elapsed time
// reflects actual GPU work without host-side noise. Essential for accurate
// kernel profiling.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#include <cuda_runtime.h>

class CUDATimer {
public:
    CUDATimer() {
        cudaEventCreate(&start_evt_);
        cudaEventCreate(&stop_evt_);
    }

    ~CUDATimer() {
        cudaEventDestroy(start_evt_);
        cudaEventDestroy(stop_evt_);
    }

    // Record start timestamp on the GPU timeline
    void start() {
        cudaEventRecord(start_evt_);
    }

    // Record stop timestamp, synchronize, and return elapsed milliseconds.
    // cudaEventSynchronize blocks the CPU until the stop event completes,
    // ensuring we capture the full GPU execution time.
    float stop() {
        cudaEventRecord(stop_evt_);
        cudaEventSynchronize(stop_evt_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_evt_, stop_evt_);
        return ms;
    }

private:
    cudaEvent_t start_evt_, stop_evt_;
};
#endif
