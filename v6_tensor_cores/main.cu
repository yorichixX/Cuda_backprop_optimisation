// ============================================================================
// V6: Tensor Core Exploitation via WMMA API
// ============================================================================
//
// THE ULTIMATE OPTIMIZATION:
//   Tensor Cores are dedicated matrix-multiply-accumulate (MMA) hardware
//   units built into NVIDIA GPUs since Volta (2017). They perform a
//   single 16×16×16 matrix multiply in ONE clock cycle — something that
//   would take hundreds of cycles on regular CUDA cores.
//
//   RTX 4060 has 96 4th-gen Tensor Cores (Ada Lovelace).
//   Peak FP16 Tensor Core throughput: ~175 TFLOPS
//   Peak FP32 CUDA Core throughput:   ~15 TFLOPS
//   That's roughly 12x more FLOPS from Tensor Cores!
//
// HOW WMMA WORKS:
//   WMMA (Warp Matrix Multiply Accumulate) is a CUDA API that exposes
//   Tensor Cores at the warp level. One warp (32 threads) cooperatively:
//     1. LOADS matrix fragments from memory into registers
//     2. COMPUTES D = A * B + C using the Tensor Core hardware
//     3. STORES the result fragment back to memory
//
//   Fragment sizes we use: 16×16×16
//     - A fragment: 16×16 (FP16)
//     - B fragment: 16×16 (FP16)
//     - C/D fragment: 16×16 (FP32 accumulator)
//
//   The 32 threads in a warp collectively hold the fragment data in their
//   registers — the programmer doesn't control which thread holds what.
//
// PADDING REQUIREMENT:
//   WMMA requires matrix dimensions to be multiples of 16.
//   Most of our layers are already multiples of 16 (128, 256, 512, 784=49×16).
//   The output layer (10 neurons) needs padding to 16.
//   Padding adds tiny overhead but enables massive throughput gain.
//
// COMPARISON TO cuBLAS:
//   cuBLAS already uses Tensor Cores internally when data is FP16 and dims
//   are multiples of 8/16. Our V1 (cuBLAS) may already benefit from this!
//   The purpose of V6 is to demonstrate EXPLICIT Tensor Core usage via WMMA,
//   showing we understand the hardware at the lowest level.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_fp16.h>
#include <mma.h>

#include "data_loader.h"
#include "mlp_config.h"
#include "timer.h"
#include "cuda_utils.cuh"

using namespace nvcuda;

// WMMA fragment dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Number of warps per block for WMMA kernels
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32
#define TILE 32  // For non-WMMA kernels

// ============================================================================
// Utility: round up to multiple of 16
// ============================================================================
inline int pad16(int x) { return ((x + 15) / 16) * 16; }

// ============================================================================
// WMMA GEMM Kernel: C(M×N) = A(M×K) * B(K×N)
// ============================================================================
// A, B in FP16 (column-major), C in FP32 (column-major)
// All dimensions must be multiples of 16!
//
// Each warp processes one 16×16 output tile.
// Multiple warps per block process adjacent tiles for better occupancy.
// ============================================================================
__global__ void wmma_gemm_nn(const __half* A, const __half* B, float* C,
                              int M, int N, int K) {
    // Which warp am I?
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int total_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    // Total number of 16×16 output tiles
    int tiles_M = M / WMMA_M;
    int tiles_N = N / WMMA_N;
    int total_tiles = tiles_M * tiles_N;

    // Each warp processes tiles in a grid-stride loop
    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += total_warps) {
        int tile_row = tile_idx % tiles_M;
        int tile_col = tile_idx / tiles_M;

        int row = tile_row * WMMA_M;
        int col = tile_col * WMMA_N;

        // Declare fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        // Initialize accumulator to zero
        wmma::fill_fragment(c_frag, 0.0f);

        // Accumulate over K dimension in steps of 16
        for (int k = 0; k < K; k += WMMA_K) {
            // Load A fragment: 16×16 sub-block starting at (row, k)
            // In column-major: ptr = A + row + k * M, leading dim = M
            const __half* a_ptr = A + row + k * M;
            wmma::load_matrix_sync(a_frag, a_ptr, M);

            // Load B fragment: 16×16 sub-block starting at (k, col)
            // In column-major: ptr = B + k + col * K, leading dim = K
            const __half* b_ptr = B + k + col * K;
            wmma::load_matrix_sync(b_frag, b_ptr, K);

            // TENSOR CORE MULTIPLY-ACCUMULATE
            // This single instruction computes a 16×16×16 matrix multiply
            // on the dedicated MMA hardware — ~12x faster than CUDA cores!
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store result fragment to global memory
        float* c_ptr = C + row + col * M;
        wmma::store_matrix_sync(c_ptr, c_frag, M, wmma::mem_col_major);
    }
}

// WMMA GEMM: C = A^T * B (A stored as K×M)
__global__ void wmma_gemm_tn(const __half* A, const __half* B, float* C,
                              int M, int N, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int total_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
    int tiles_M = M / WMMA_M, tiles_N = N / WMMA_N;

    for (int ti = warp_id; ti < tiles_M * tiles_N; ti += total_warps) {
        int tr = ti % tiles_M, tc = ti / tiles_M;
        int row = tr * WMMA_M, col = tc * WMMA_N;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k = 0; k < K; k += WMMA_K) {
            // A^T[row, k] → A[k, row], stored as K×M col-major: ptr = A + k + row*K
            // Load as row_major with leading dim K
            wmma::load_matrix_sync(a_frag, A + k + row * K, K);
            wmma::load_matrix_sync(b_frag, B + k + col * K, K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(C + row + col * M, c_frag, M, wmma::mem_col_major);
    }
}

// ============================================================================
// Element-wise kernels (FP16 activations, FP32 Z/delta)
// ============================================================================

__global__ void add_bias_sigmoid_kernel(const float* Z_in, const float* bias,
                                         float* Z_out, __half* A_out, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int row = idx % M;
        float z = Z_in[idx] + bias[row];
        Z_out[idx] = z;
        z = fmaxf(-30.0f, fminf(30.0f, z));
        A_out[idx] = __float2half(1.0f / (1.0f + expf(-z)));
    }
}

__global__ void add_bias_relu_kernel(const float* Z_in, const float* bias,
                                      float* Z_out, __half* A_out, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int row = idx % M;
        float z = Z_in[idx] + bias[row];
        Z_out[idx] = z;
        A_out[idx] = __float2half(fmaxf(0.0f, z));
    }
}

__global__ void float2half_kernel(const float* src, __half* dst, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}
__global__ void half2float_kernel(const __half* src, float* dst, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) dst[i] = __half2float(src[i]);
}

__global__ void output_delta_sigmoid_fp16(float* delta, const __half* output,
                                           const float* target, int N, float inv_B) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) { float o = __half2float(output[i]); delta[i] = (o-target[i])*o*(1-o)*inv_B; }
}
__global__ void sigmoid_backward_fp16(float* delta, const __half* a, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) { float ai = __half2float(a[i]); delta[i] *= ai*(1-ai); }
}
__global__ void relu_backward_fp32(float* delta, const float* z, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) delta[i] *= (z[i]>0)?1.0f:0.0f;
}
__global__ void bias_update_kernel(float* bias, const float* delta, float lr, int M, int B) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < M) { float g=0; for(int j=0;j<B;j++) g+=delta[i+j*M]; bias[i]-=lr*g; }
}

// FP32 tiled GEMM for backward pass (delta is FP32)
__global__ void tiled_gemm_tn_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tA[TILE][TILE+1], tB[TILE][TILE+1];
    int r=blockIdx.y*TILE+threadIdx.y, c=blockIdx.x*TILE+threadIdx.x;
    float sum=0;
    for(int t=0;t<div_ceil(K,TILE);t++){
        int ak=t*TILE+threadIdx.x;
        tA[threadIdx.y][threadIdx.x]=(r<M&&ak<K)?A[ak+r*K]:0;
        int br=t*TILE+threadIdx.y;
        tB[threadIdx.y][threadIdx.x]=(br<K&&c<N)?B[br+c*K]:0;
        __syncthreads();
        for(int i=0;i<TILE;i++) sum+=tA[threadIdx.y][i]*tB[i][threadIdx.x];
        __syncthreads();
    }
    if(r<M&&c<N) C[r+c*M]=sum;
}
__global__ void tiled_gemm_nt_update_fp32(const float* A, const float* B, float* C,
                                           int M, int N, int K, float alpha, float bv) {
    __shared__ float tA[TILE][TILE+1], tB[TILE][TILE+1];
    int r=blockIdx.y*TILE+threadIdx.y, c=blockIdx.x*TILE+threadIdx.x;
    float sum=0;
    for(int t=0;t<div_ceil(K,TILE);t++){
        int ac=t*TILE+threadIdx.x;
        tA[threadIdx.y][threadIdx.x]=(r<M&&ac<K)?A[r+ac*M]:0;
        int bk=t*TILE+threadIdx.y;
        tB[threadIdx.y][threadIdx.x]=(c<N&&bk<K)?B[c+bk*N]:0;
        __syncthreads();
        for(int i=0;i<TILE;i++) sum+=tA[threadIdx.y][i]*tB[i][threadIdx.x];
        __syncthreads();
    }
    if(r<M&&c<N){int idx=r+c*M; C[idx]=bv*C[idx]+alpha*sum;}
}

// ============================================================================
// MLP with Tensor Cores
// ============================================================================

struct MLP_GPU {
    MLPConfig config;
    int num_layers;

    // Padded layer sizes (multiples of 16 for WMMA)
    std::vector<int> padded_sizes;

    std::vector<float*> d_weights_fp32;   // Master weights (FP32), padded
    std::vector<__half*> d_weights_fp16;  // Compute weights (FP16), padded
    std::vector<float*> d_biases;         // Padded

    std::vector<__half*> d_a_fp16;        // Activations (FP16), padded
    std::vector<float*> d_a_fp32;         // FP32 scratch for activations (weight update)
    std::vector<float*> d_z;              // Pre-activations (FP32), padded
    std::vector<float*> d_delta;          // Deltas (FP32), padded

    // Padded training data
    __half* d_train_images_fp16;
    float*  d_train_images_fp32;  // Keep FP32 copy for weight updates
    float*  d_train_labels;
    float*  d_train_labels_padded;  // Padded to output dim multiple of 16
    int     num_train_samples;

    float* d_loss;
    float* d_z_gemm_out;  // Temp buffer for WMMA GEMM output

    MLP_GPU(const MLPConfig& cfg, const MNISTData& td) : config(cfg) {
        num_layers = config.num_weight_layers();
        int B = config.batch_size;
        num_train_samples = td.num_samples;

        // Compute padded sizes
        padded_sizes.resize(config.layer_sizes.size());
        for (size_t i = 0; i < config.layer_sizes.size(); i++)
            padded_sizes[i] = pad16(config.layer_sizes[i]);

        std::mt19937 rng(config.seed);
        d_weights_fp32.resize(num_layers); d_weights_fp16.resize(num_layers);
        d_biases.resize(num_layers);
        d_z.resize(num_layers); d_delta.resize(num_layers);
        d_a_fp16.resize(num_layers + 1);

        for (int l = 0; l < num_layers; l++) {
            int ni = padded_sizes[l], no = padded_sizes[l+1];
            int ni_orig = config.layer_sizes[l], no_orig = config.layer_sizes[l+1];

            // Initialize with orig dims, pad with zeros
            float lim = std::sqrt(6.0f / (ni_orig + no_orig));
            std::uniform_real_distribution<float> dist(-lim, lim);
            std::vector<float> hw(no * ni, 0.0f);
            for (int j = 0; j < ni_orig; j++)
                for (int i = 0; i < no_orig; i++)
                    hw[i + j * no] = dist(rng);

            CUDA_CHECK(cudaMalloc(&d_weights_fp32[l], no*ni*sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights_fp32[l], hw.data(), no*ni*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_weights_fp16[l], no*ni*sizeof(__half)));
            float2half_kernel<<<div_ceil(no*ni,BLOCK_SIZE),BLOCK_SIZE>>>(d_weights_fp32[l], d_weights_fp16[l], no*ni);

            CUDA_CHECK(cudaMalloc(&d_biases[l], no*sizeof(float)));
            CUDA_CHECK(cudaMemset(d_biases[l], 0, no*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_z[l], no*B*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_delta[l], no*B*sizeof(float)));
        }

        d_a_fp16[0] = nullptr;
        for (int l = 1; l <= num_layers; l++)
            CUDA_CHECK(cudaMalloc(&d_a_fp16[l], padded_sizes[l]*B*sizeof(__half)));

        // FP32 activation scratch for weight updates
        d_a_fp32.resize(num_layers + 1);
        for (int l = 0; l <= num_layers; l++)
            CUDA_CHECK(cudaMalloc(&d_a_fp32[l], padded_sizes[l]*B*sizeof(float)));

        // Upload and pad training data
        int id = config.layer_sizes[0], id_p = padded_sizes[0];
        int od = config.layer_sizes.back(), od_p = padded_sizes.back();
        int ns = td.num_samples;

        // Images: pad each sample from id to id_p (784 is already 16-aligned)
        CUDA_CHECK(cudaMalloc(&d_train_images_fp32, ns*id_p*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_train_images_fp32, 0, ns*id_p*sizeof(float)));
        if (id == id_p) {
            CUDA_CHECK(cudaMemcpy(d_train_images_fp32, td.images.data(), ns*id*sizeof(float), cudaMemcpyHostToDevice));
        } else {
            for (int s = 0; s < ns; s++)
                CUDA_CHECK(cudaMemcpy(d_train_images_fp32 + s*id_p, td.get_image(s), id*sizeof(float), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc(&d_train_images_fp16, ns*id_p*sizeof(__half)));
        float2half_kernel<<<div_ceil(ns*id_p,BLOCK_SIZE),BLOCK_SIZE>>>(d_train_images_fp32, d_train_images_fp16, ns*id_p);

        // Labels: pad from od to od_p
        CUDA_CHECK(cudaMalloc(&d_train_labels, ns*od*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_train_labels, td.labels.data(), ns*od*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_train_labels_padded, ns*od_p*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_train_labels_padded, 0, ns*od_p*sizeof(float)));
        if (od == od_p) {
            CUDA_CHECK(cudaMemcpy(d_train_labels_padded, td.labels.data(), ns*od*sizeof(float), cudaMemcpyHostToDevice));
        } else {
            for (int s = 0; s < ns; s++)
                CUDA_CHECK(cudaMemcpy(d_train_labels_padded + s*od_p, td.get_label(s), od*sizeof(float), cudaMemcpyHostToDevice));
        }

        // Temp buffer for GEMM output
        int max_no = *std::max_element(padded_sizes.begin()+1, padded_sizes.end());
        CUDA_CHECK(cudaMalloc(&d_z_gemm_out, max_no*B*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~MLP_GPU() {
        for(int l=0;l<num_layers;l++){
            cudaFree(d_weights_fp32[l]); cudaFree(d_weights_fp16[l]);
            cudaFree(d_biases[l]); cudaFree(d_z[l]); cudaFree(d_delta[l]);
        }
        for(int l=1;l<=num_layers;l++) cudaFree(d_a_fp16[l]);
        for(int l=0;l<=num_layers;l++) cudaFree(d_a_fp32[l]);
        cudaFree(d_train_images_fp16); cudaFree(d_train_images_fp32);
        cudaFree(d_train_labels); cudaFree(d_train_labels_padded);
        cudaFree(d_z_gemm_out); cudaFree(d_loss);
    }

    void forward(int batch_offset, int B) {
        int id_p = padded_sizes[0];
        d_a_fp16[0] = d_train_images_fp16 + batch_offset * id_p;

        for (int l = 0; l < num_layers; l++) {
            int ni = padded_sizes[l], no = padded_sizes[l+1];
            int total = no * B;
            bool io = (l == num_layers-1);

            // WMMA GEMM: Z_raw = W * A  (padded dims, all multiples of 16)
            int total_warps_needed = (no / WMMA_M) * (B / WMMA_N);
            // Make sure B is a multiple of 16 for WMMA
            // If not, fall back to padded B
            if (B % 16 != 0) {
                // Fall back: use the tiled GEMM approach for non-aligned batch sizes
                // For simplicity, we'll just launch WMMA with the padded assumption
                // In production, you'd pad the batch too
            }

            // Launch WMMA kernel
            int warps_needed = (no / WMMA_M) * (B >= 16 ? B / WMMA_N : 1);
            int threads = WARPS_PER_BLOCK * WARP_SIZE;
            int blocks = div_ceil(warps_needed, WARPS_PER_BLOCK);
            if (blocks > 0 && B >= 16) {
                wmma_gemm_nn<<<blocks, threads>>>(
                    d_weights_fp16[l], d_a_fp16[l], d_z_gemm_out, no, B, ni);
            }

            // Add bias + activation (element-wise, not WMMA)
            int g1 = div_ceil(total, BLOCK_SIZE);
            if (io || config.hidden_activation == ActivationType::SIGMOID)
                add_bias_sigmoid_kernel<<<g1, BLOCK_SIZE>>>(d_z_gemm_out, d_biases[l], d_z[l], d_a_fp16[l+1], no, B);
            else
                add_bias_relu_kernel<<<g1, BLOCK_SIZE>>>(d_z_gemm_out, d_biases[l], d_z[l], d_a_fp16[l+1], no, B);
        }
    }

    void backward_and_update(int batch_offset, int B) {
        float lr = config.learning_rate, inv_B = 1.0f/B;
        int od_p = padded_sizes.back();
        float* d_target = d_train_labels_padded + batch_offset * od_p;

        // Output delta
        output_delta_sigmoid_fp16<<<div_ceil(od_p*B,BLOCK_SIZE),BLOCK_SIZE>>>(
            d_delta[num_layers-1], d_a_fp16[num_layers], d_target, od_p*B, inv_B);

        // Hidden deltas (FP32 backward pass for stability)
        for (int l = num_layers-2; l >= 0; l--) {
            int nt = padded_sizes[l+1], nn = padded_sizes[l+2];
            dim3 bk(TILE,TILE), gg(div_ceil(B,TILE),div_ceil(nt,TILE));
            tiled_gemm_tn_fp32<<<gg,bk>>>(d_weights_fp32[l+1],d_delta[l+1],d_delta[l],nt,B,nn);
            int g1 = div_ceil(nt*B,BLOCK_SIZE);
            if (config.hidden_activation == ActivationType::SIGMOID)
                sigmoid_backward_fp16<<<g1,BLOCK_SIZE>>>(d_delta[l],d_a_fp16[l+1],nt*B);
            else
                relu_backward_fp32<<<g1,BLOCK_SIZE>>>(d_delta[l],d_z[l],nt*B);
        }

        // Convert FP16 activations to FP32 for weight update
        {
            int id_p = padded_sizes[0];
            CUDA_CHECK(cudaMemcpy(d_a_fp32[0], d_train_images_fp32 + batch_offset * id_p,
                                  id_p * B * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        for (int l = 1; l <= num_layers; l++) {
            int n = padded_sizes[l];
            half2float_kernel<<<div_ceil(n*B,BLOCK_SIZE),BLOCK_SIZE>>>(d_a_fp16[l], d_a_fp32[l], n*B);
        }

        // Weight update (FP32 masters)
        for (int l = 0; l < num_layers; l++) {
            int ni = padded_sizes[l], no = padded_sizes[l+1];
            dim3 bk(TILE,TILE), gg(div_ceil(ni,TILE),div_ceil(no,TILE));
            // W -= lr * delta[l] * a[l]^T
            tiled_gemm_nt_update_fp32<<<gg,bk>>>(d_delta[l],d_a_fp32[l],d_weights_fp32[l],no,ni,B,-lr,1.0f);
            bias_update_kernel<<<div_ceil(no,BLOCK_SIZE),BLOCK_SIZE>>>(d_biases[l],d_delta[l],lr,no,B);
            // Re-sync FP16 weights
            float2half_kernel<<<div_ceil(no*ni,BLOCK_SIZE),BLOCK_SIZE>>>(d_weights_fp32[l],d_weights_fp16[l],no*ni);
        }
    }

    float train_batch(int batch_start, int B) {
        forward(batch_start, B);
        int od_p = padded_sizes.back();
        float* d_output_fp32;
        CUDA_CHECK(cudaMalloc(&d_output_fp32, od_p*B*sizeof(float)));
        half2float_kernel<<<div_ceil(od_p*B,BLOCK_SIZE),BLOCK_SIZE>>>(d_a_fp16[num_layers],d_output_fp32,od_p*B);
        float loss = compute_mse_loss(d_output_fp32, d_train_labels_padded+batch_start*od_p, od_p*B, d_loss);
        cudaFree(d_output_fp32);
        backward_and_update(batch_start, B);
        return loss;
    }

    float evaluate(const MNISTData& td) {
        int id = config.layer_sizes[0], od = config.layer_sizes.back();
        int id_p = padded_sizes[0], od_p = padded_sizes.back();
        int B = config.batch_size, correct = 0;

        // Ensure batch size is multiple of 16 for WMMA
        int eval_B = (B / 16) * 16;
        if (eval_B == 0) eval_B = 16;

        float* dt_fp32; __half* dt_fp16;
        CUDA_CHECK(cudaMalloc(&dt_fp32, id_p*eval_B*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dt_fp16, id_p*eval_B*sizeof(__half)));
        std::vector<float> ho(od_p*eval_B);

        for (int s = 0; s < td.num_samples; s += eval_B) {
            int aB = std::min(eval_B, td.num_samples - s);
            if (aB % 16 != 0) aB = (aB / 16) * 16;  // Ensure alignment
            if (aB <= 0) break;

            CUDA_CHECK(cudaMemset(dt_fp32, 0, id_p*aB*sizeof(float)));
            // Copy images (may need padding per sample)
            if (id == id_p) {
                CUDA_CHECK(cudaMemcpy(dt_fp32, td.get_image(s), id*aB*sizeof(float), cudaMemcpyHostToDevice));
            } else {
                for (int i = 0; i < aB; i++)
                    CUDA_CHECK(cudaMemcpy(dt_fp32+i*id_p, td.get_image(s+i), id*sizeof(float), cudaMemcpyHostToDevice));
            }
            float2half_kernel<<<div_ceil(id_p*aB,BLOCK_SIZE),BLOCK_SIZE>>>(dt_fp32,dt_fp16,id_p*aB);

            __half* sv = d_a_fp16[0]; d_a_fp16[0] = dt_fp16;

            for (int l = 0; l < num_layers; l++) {
                int ni = padded_sizes[l], no = padded_sizes[l+1];
                int warps = (no/WMMA_M)*(aB/WMMA_N);
                int threads = WARPS_PER_BLOCK*WARP_SIZE;
                int blocks = div_ceil(warps, WARPS_PER_BLOCK);
                if (blocks > 0)
                    wmma_gemm_nn<<<blocks,threads>>>(d_weights_fp16[l],d_a_fp16[l],d_z_gemm_out,no,aB,ni);
                int g1 = div_ceil(no*aB,BLOCK_SIZE);
                bool io = (l==num_layers-1);
                if (io||config.hidden_activation==ActivationType::SIGMOID)
                    add_bias_sigmoid_kernel<<<g1,BLOCK_SIZE>>>(d_z_gemm_out,d_biases[l],d_z[l],d_a_fp16[l+1],no,aB);
                else
                    add_bias_relu_kernel<<<g1,BLOCK_SIZE>>>(d_z_gemm_out,d_biases[l],d_z[l],d_a_fp16[l+1],no,aB);
            }

            float* d_out_fp32;
            CUDA_CHECK(cudaMalloc(&d_out_fp32,od_p*aB*sizeof(float)));
            half2float_kernel<<<div_ceil(od_p*aB,BLOCK_SIZE),BLOCK_SIZE>>>(d_a_fp16[num_layers],d_out_fp32,od_p*aB);
            CUDA_CHECK(cudaMemcpy(ho.data(),d_out_fp32,od_p*aB*sizeof(float),cudaMemcpyDeviceToHost));
            cudaFree(d_out_fp32);

            for (int i = 0; i < aB; i++) {
                int p=0,a=0; float mp=-1e30f,ml=-1e30f;
                for (int c = 0; c < od; c++) {
                    if(ho[c+i*od_p]>mp){mp=ho[c+i*od_p];p=c;}
                    if(td.labels[(s+i)*od+c]>ml){ml=td.labels[(s+i)*od+c];a=c;}
                }
                if(p==a) correct++;
            }
            d_a_fp16[0] = sv;
        }
        cudaFree(dt_fp32); cudaFree(dt_fp16);
        return (float)correct / td.num_samples;
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    printf("================================================================\n");
    printf("  V6: Tensor Cores via WMMA API\n");
    printf("================================================================\n");

    MLPConfig config = parse_args(argc, argv);

    // Ensure batch size is multiple of 16 for WMMA
    if (config.batch_size % 16 != 0) {
        config.batch_size = (config.batch_size / 16) * 16;
        if (config.batch_size == 0) config.batch_size = 16;
        printf("Note: Adjusted batch_size to %d (must be multiple of 16 for WMMA)\n",
               config.batch_size);
    }
    config.print();

    MNISTData train = load_mnist(config.data_dir+"/train-images-idx3-ubyte",
                                  config.data_dir+"/train-labels-idx1-ubyte");
    MNISTData test = load_mnist(config.data_dir+"/t10k-images-idx3-ubyte",
                                 config.data_dir+"/t10k-labels-idx1-ubyte");

    if (config.layer_sizes.front()!=train.image_size||config.layer_sizes.back()!=train.num_classes){
        fprintf(stderr,"Error: Architecture mismatch\n"); return 1;
    }

    MLP_GPU mlp(config, train);
    int nb = train.num_samples / config.batch_size;
    double tt = 0;

    printf("\nTraining: %d samples, %d batches/epoch (batch=%d)\n\n",
           train.num_samples, nb, config.batch_size);
    printf("Padded layer sizes: ");
    for (auto s : mlp.padded_sizes) printf("%d ", s);
    printf("\n\n");

    mlp.forward(0, config.batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CPUTimer timer; CUDATimer ctimer;
    for (int ep = 0; ep < config.epochs; ep++) {
        timer.start(); ctimer.start();
        float loss = 0;
        for (int b = 0; b < nb; b++)
            loss += mlp.train_batch(b*config.batch_size, config.batch_size);
        float gt = ctimer.stop(); double wt = timer.elapsed_ms(); tt += wt;
        float acc = mlp.evaluate(test);
        printf("Epoch %2d/%d | Loss: %.6f | Accuracy: %6.2f%% | GPU: %8.1f ms | Wall: %8.1f ms\n",
               ep+1, config.epochs, loss/nb, acc*100, gt, wt);
    }

    printf("\n================================================================\n");
    printf("  Results — V6 Tensor Cores (WMMA)\n");
    printf("================================================================\n");
    printf("Total training time:   %10.1f ms\n", tt);
    printf("Average epoch time:    %10.1f ms\n", tt / config.epochs);
    printf("================================================================\n");
    return 0;
}
