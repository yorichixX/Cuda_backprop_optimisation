cmake_minimum_required(VERSION 3.20)
project(CudaBackProp LANGUAGES CXX CUDA)

# ============================================================================
# Build Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Target Ada Lovelace (RTX 4060) — SM 8.9
# This tells nvcc to generate code specifically optimized for our GPU.
# Using the exact SM version avoids JIT compilation overhead at runtime.
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find the CUDA Toolkit (provides CUDA::cublas, CUDA::cudart, etc.)
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Common Library (pure C++ — no CUDA dependency)
# ============================================================================
# This library contains data loading, config parsing, and CPU timing utilities.
# It is compiled with the host C++ compiler, not nvcc. Every version links to it.
add_library(common STATIC common/data_loader.cpp)
target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR}/common)

# ============================================================================
# V0: CPU Baseline — Sequential single-threaded C++
# ============================================================================
# Replicates the "sequential CPU" baseline from Sierra-Canto et al. (2010).
# No CUDA, no parallelism — this is the reference that everything beats.
add_executable(v0_cpu
    v0_cpu_baseline/main.cpp
    v0_cpu_baseline/mlp_cpu.cpp)
target_link_libraries(v0_cpu PRIVATE common)

# ============================================================================
# V1: Naive CUDA — cuBLAS + Custom Kernels (The Paper's Approach)
# ============================================================================
# Uses cuBLAS for GEMM and hand-written kernels for element-wise ops.
# This is what Sierra-Canto et al. actually implemented.
add_executable(v1_naive v1_naive_cuda/main.cu)
target_link_libraries(v1_naive PRIVATE common CUDA::cublas)

# ============================================================================
# V2: Custom Tiled GEMM with Shared Memory
# ============================================================================
# Replaces cuBLAS with our own tiled matrix multiply kernel.
# Demonstrates shared memory tiling and bank conflict avoidance.
add_executable(v2_tiled v2_tiled_gemm/main.cu)
target_link_libraries(v2_tiled PRIVATE common)

# ============================================================================
# V3: Fused Kernels — GEMM + Bias + Activation in One Launch
# ============================================================================
# Eliminates intermediate global memory writes by fusing operations.
# Key optimization: write to global memory only once per layer.
add_executable(v3_fused v3_fused_kernels/main.cu)
target_link_libraries(v3_fused PRIVATE common)

# ============================================================================
# V4: CUDA Streams + Pinned Memory — Overlap Transfer & Compute
# ============================================================================
# Demonstrates asynchronous data transfers overlapped with computation.
# Uses double-buffering and pinned (page-locked) host memory.
add_executable(v4_streams v4_streams/main.cu)
target_link_libraries(v4_streams PRIVATE common)

# ============================================================================
# V5: Mixed Precision — FP16 Storage, FP32 Accumulation
# ============================================================================
# Halves memory bandwidth requirements by storing weights/activations in FP16.
# Maintains numerical stability via FP32 accumulation and master weights.
add_executable(v5_mixed v5_mixed_precision/main.cu)
target_link_libraries(v5_mixed PRIVATE common)

# ============================================================================
# V6: Tensor Cores via WMMA — Hardware-Accelerated Matrix Multiply
# ============================================================================
# Uses the 96 4th-gen Tensor Cores on the RTX 4060 via the WMMA API.
# Peak throughput path: FP16 inputs → FP32 accumulation → hardware MMA.
add_executable(v6_tensor v6_tensor_cores/main.cu)
target_link_libraries(v6_tensor PRIVATE common)
