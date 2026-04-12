// ============================================================================
// V5: Mixed Precision — FP16 Storage, FP32 Accumulation
// ============================================================================
//
// KEY IMPROVEMENT:
//   Store weights and activations in FP16 (half precision, 2 bytes) instead
//   of FP32 (single precision, 4 bytes). This provides:
//
//   1. 2x LESS MEMORY BANDWIDTH consumed per element
//      → Memory-bound kernels run up to 2x faster
//      → More data fits in cache/shared memory
//
//   2. 2x MORE DATA per shared memory tile
//      → Could use larger tiles, or same tiles with less shared memory pressure
//
//   3. NATIVE FP16 THROUGHPUT on Ada Lovelace
//      → RTX 4060 has 2x FP16 throughput vs FP32 for element-wise ops
//
// NUMERICAL STABILITY:
//   Pure FP16 has only ~3 decimal digits of precision (vs ~7 for FP32).
//   This causes problems during training:
//     - Small gradients underflow to zero → learning stops
//     - Weight updates (large_weight - small_gradient) lose precision
//
//   Solution: MIXED PRECISION pattern
//     - COMPUTE: FP16 for GEMM inputs (weights, activations)
//     - ACCUMULATE: FP32 for dot product sums (prevents rounding errors)
//     - MASTER WEIGHTS: FP32 copy of weights for SGD updates
//     - After update: convert FP32 master weights → FP16 for next forward pass
//
//   This gives us FP16 speed with FP32 accuracy.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_fp16.h>

#include "data_loader.h"
#include "mlp_config.h"
#include "timer.h"
#include "cuda_utils.cuh"

#define TILE 32

// ============================================================================
// Conversion kernels
// ============================================================================

__global__ void float2half_kernel(const float* src, __half* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}

__global__ void half2float_kernel(const __half* src, float* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __half2float(src[i]);
}

// ============================================================================
// Mixed-Precision Tiled GEMM: FP16 inputs, FP32 accumulation
// ============================================================================
// Loads tiles as FP16 (half the bandwidth!), converts to FP32 for the
// multiply-accumulate. Final result stored as FP32.
//
// This is the KEY advantage: same shared memory holds 2x more data in FP16,
// and the loads from global memory consume half the bandwidth.
// ============================================================================

// C_fp32(M×N) = A_fp16(M×K) * B_fp16(K×N), accumulated in FP32
__global__ void mixed_gemm_nn(const __half* A, const __half* B, float* C,
                               int M, int N, int K) {
    __shared__ __half tA[TILE][TILE+1];
    __shared__ __half tB[TILE][TILE+1];

    int r = blockIdx.y*TILE+threadIdx.y, c = blockIdx.x*TILE+threadIdx.x;
    float sum = 0.0f;  // FP32 accumulator — the key to numerical stability

    for (int t = 0; t < div_ceil(K, TILE); t++) {
        int ac = t*TILE+threadIdx.x;
        tA[threadIdx.y][threadIdx.x] = (r<M && ac<K) ? A[r+ac*M] : __float2half(0.0f);
        int br = t*TILE+threadIdx.y;
        tB[threadIdx.y][threadIdx.x] = (br<K && c<N) ? B[br+c*K] : __float2half(0.0f);
        __syncthreads();
        for (int i = 0; i < TILE; i++)
            sum += __half2float(tA[threadIdx.y][i]) * __half2float(tB[i][threadIdx.x]);
        __syncthreads();
    }
    if (r<M && c<N) C[r+c*M] = sum;
}

// C_fp32(M×N) = A_fp16^T * B_fp16  (A stored as K×M)
__global__ void mixed_gemm_tn(const __half* A, const __half* B, float* C,
                               int M, int N, int K) {
    __shared__ __half tA[TILE][TILE+1], tB[TILE][TILE+1];
    int r=blockIdx.y*TILE+threadIdx.y, c=blockIdx.x*TILE+threadIdx.x;
    float sum=0;
    for(int t=0;t<div_ceil(K,TILE);t++){
        int ak=t*TILE+threadIdx.x;
        tA[threadIdx.y][threadIdx.x]=(r<M&&ak<K)?A[ak+r*K]:__float2half(0.0f);
        int br=t*TILE+threadIdx.y;
        tB[threadIdx.y][threadIdx.x]=(br<K&&c<N)?B[br+c*K]:__float2half(0.0f);
        __syncthreads();
        for(int i=0;i<TILE;i++) sum+=__half2float(tA[threadIdx.y][i])*__half2float(tB[i][threadIdx.x]);
        __syncthreads();
    }
    if(r<M&&c<N) C[r+c*M]=sum;
}

// C_fp32 = alpha * A_fp16 * B_fp16^T + beta * C_fp32 (for weight update)
__global__ void mixed_gemm_nt_update(const __half* A, const __half* B, float* C,
                                      int M, int N, int K, float alpha, float bv) {
    __shared__ __half tA[TILE][TILE+1], tB[TILE][TILE+1];
    int r=blockIdx.y*TILE+threadIdx.y, c=blockIdx.x*TILE+threadIdx.x;
    float sum=0;
    for(int t=0;t<div_ceil(K,TILE);t++){
        int ac=t*TILE+threadIdx.x;
        tA[threadIdx.y][threadIdx.x]=(r<M&&ac<K)?A[r+ac*M]:__float2half(0.0f);
        int bk=t*TILE+threadIdx.y;
        tB[threadIdx.y][threadIdx.x]=(c<N&&bk<K)?B[c+bk*N]:__float2half(0.0f);
        __syncthreads();
        for(int i=0;i<TILE;i++) sum+=__half2float(tA[threadIdx.y][i])*__half2float(tB[i][threadIdx.x]);
        __syncthreads();
    }
    if(r<M&&c<N){int idx=r+c*M; C[idx]=bv*C[idx]+alpha*sum;}
}

// ============================================================================
// Fused forward kernels with FP16 I/O
// ============================================================================
// Inputs W, X in FP16. Bias in FP32. Outputs Z (FP32) and A (FP16).
// Z is kept FP32 for backprop stability.

__global__ void fused_mixed_sigmoid(
        const __half* W, const __half* X, const float* bias,
        float* Z, __half* A, int M, int N, int K) {
    __shared__ __half tW[TILE][TILE+1], tX[TILE][TILE+1];
    int r=blockIdx.y*TILE+threadIdx.y, c=blockIdx.x*TILE+threadIdx.x;
    float sum=0;
    for(int t=0;t<div_ceil(K,TILE);t++){
        int wc=t*TILE+threadIdx.x;
        tW[threadIdx.y][threadIdx.x]=(r<M&&wc<K)?W[r+wc*M]:__float2half(0.0f);
        int xr=t*TILE+threadIdx.y;
        tX[threadIdx.y][threadIdx.x]=(xr<K&&c<N)?X[xr+c*K]:__float2half(0.0f);
        __syncthreads();
        for(int i=0;i<TILE;i++) sum+=__half2float(tW[threadIdx.y][i])*__half2float(tX[i][threadIdx.x]);
        __syncthreads();
    }
    if(r<M&&c<N){
        int idx=r+c*M;
        float z=sum+bias[r]; Z[idx]=z;
        z=fmaxf(-30.0f,fminf(30.0f,z));
        A[idx]=__float2half(1.0f/(1.0f+expf(-z)));
    }
}

__global__ void fused_mixed_relu(
        const __half* W, const __half* X, const float* bias,
        float* Z, __half* A, int M, int N, int K) {
    __shared__ __half tW[TILE][TILE+1], tX[TILE][TILE+1];
    int r=blockIdx.y*TILE+threadIdx.y, c=blockIdx.x*TILE+threadIdx.x;
    float sum=0;
    for(int t=0;t<div_ceil(K,TILE);t++){
        int wc=t*TILE+threadIdx.x;
        tW[threadIdx.y][threadIdx.x]=(r<M&&wc<K)?W[r+wc*M]:__float2half(0.0f);
        int xr=t*TILE+threadIdx.y;
        tX[threadIdx.y][threadIdx.x]=(xr<K&&c<N)?X[xr+c*K]:__float2half(0.0f);
        __syncthreads();
        for(int i=0;i<TILE;i++) sum+=__half2float(tW[threadIdx.y][i])*__half2float(tX[i][threadIdx.x]);
        __syncthreads();
    }
    if(r<M&&c<N){
        int idx=r+c*M; float z=sum+bias[r];
        Z[idx]=z; A[idx]=__float2half(fmaxf(0.0f,z));
    }
}

// ============================================================================
// Backward element-wise kernels (FP32 deltas, FP16 activations)
// ============================================================================

__global__ void output_delta_sigmoid_fp16(
        float* delta, const __half* output, const float* target,
        int N, float inv_B) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float o=__half2float(output[i]); delta[i]=(o-target[i])*o*(1-o)*inv_B;}
}

// delta_fp32 *= sigmoid'(a_fp16)
__global__ void sigmoid_backward_fp16(float* delta, const __half* a, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float ai=__half2float(a[i]); delta[i]*=ai*(1-ai);}
}

__global__ void relu_backward_fp32(float* delta, const float* z, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) delta[i]*=(z[i]>0)?1.0f:0.0f;
}

__global__ void bias_update_fp32(float* bias, const float* delta, float lr, int M, int B) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<M){float g=0; for(int j=0;j<B;j++) g+=delta[i+j*M]; bias[i]-=lr*g;}
}

// Backward GEMM: delta_fp32 uses FP32 for all backward computations
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
// MLP with Mixed Precision
// ============================================================================

struct MLP_GPU {
    MLPConfig config;
    int num_layers;

    // FP32 master weights (for SGD updates — full precision)
    std::vector<float*> d_weights_fp32;
    // FP16 weights (for forward/backward compute — half bandwidth)
    std::vector<__half*> d_weights_fp16;
    // Biases stay FP32 (small, not bandwidth-limited)
    std::vector<float*> d_biases;

    // Activations in FP16 (reduced memory footprint)
    std::vector<__half*> d_a_fp16;  // a[0..L]
    // FP32 scratch buffers for activations (needed during weight update GEMM)
    std::vector<float*> d_a_fp32;  // a[0..L] in FP32, allocated per max batch
    // Pre-activations and deltas in FP32 (for numerical stability in backprop)
    std::vector<float*> d_z, d_delta;

    float *d_train_images, *d_train_labels;
    __half* d_train_images_fp16;
    float* d_loss;

    MLP_GPU(const MLPConfig& cfg, const MNISTData& td) : config(cfg) {
        num_layers = config.num_weight_layers();
        int B = config.batch_size;
        std::mt19937 rng(config.seed);

        d_weights_fp32.resize(num_layers); d_weights_fp16.resize(num_layers);
        d_biases.resize(num_layers);
        d_z.resize(num_layers); d_delta.resize(num_layers);
        d_a_fp16.resize(num_layers + 1);

        for (int l = 0; l < num_layers; l++) {
            int ni=config.layer_sizes[l], no=config.layer_sizes[l+1];
            float lim=std::sqrt(6.0f/(ni+no));
            std::uniform_real_distribution<float> dist(-lim,lim);
            std::vector<float> hw(no*ni);
            for(auto& w:hw) w=dist(rng);

            // Master weights (FP32)
            CUDA_CHECK(cudaMalloc(&d_weights_fp32[l], no*ni*sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights_fp32[l], hw.data(), no*ni*sizeof(float), cudaMemcpyHostToDevice));
            // FP16 copy
            CUDA_CHECK(cudaMalloc(&d_weights_fp16[l], no*ni*sizeof(__half)));
            float2half_kernel<<<div_ceil(no*ni,BLOCK_SIZE),BLOCK_SIZE>>>(d_weights_fp32[l], d_weights_fp16[l], no*ni);

            CUDA_CHECK(cudaMalloc(&d_biases[l], no*sizeof(float)));
            CUDA_CHECK(cudaMemset(d_biases[l], 0, no*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_z[l], no*B*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_delta[l], no*B*sizeof(float)));
        }

        d_a_fp16[0] = nullptr;
        for (int l = 1; l <= num_layers; l++)
            CUDA_CHECK(cudaMalloc(&d_a_fp16[l], config.layer_sizes[l]*B*sizeof(__half)));

        // FP32 activation scratch for weight updates
        d_a_fp32.resize(num_layers + 1);
        for (int l = 0; l <= num_layers; l++)
            CUDA_CHECK(cudaMalloc(&d_a_fp32[l], config.layer_sizes[l]*B*sizeof(float)));

        // Upload training data + convert to FP16
        int ns=td.num_samples, id=config.layer_sizes[0], od=config.layer_sizes.back();
        CUDA_CHECK(cudaMalloc(&d_train_images, ns*id*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_train_images_fp16, ns*id*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_train_labels, ns*od*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_train_images, td.images.data(), ns*id*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_train_labels, td.labels.data(), ns*od*sizeof(float), cudaMemcpyHostToDevice));
        float2half_kernel<<<div_ceil(ns*id,BLOCK_SIZE),BLOCK_SIZE>>>(d_train_images, d_train_images_fp16, ns*id);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    }

    ~MLP_GPU() {
        for(int l=0;l<num_layers;l++){
            cudaFree(d_weights_fp32[l]); cudaFree(d_weights_fp16[l]);
            cudaFree(d_biases[l]); cudaFree(d_z[l]); cudaFree(d_delta[l]);
        }
        for(int l=1;l<=num_layers;l++) cudaFree(d_a_fp16[l]);
        for(int l=0;l<=num_layers;l++) cudaFree(d_a_fp32[l]);
        cudaFree(d_train_images); cudaFree(d_train_images_fp16);
        cudaFree(d_train_labels); cudaFree(d_loss);
    }

    void forward(int batch_offset, int B) {
        d_a_fp16[0] = d_train_images_fp16 + batch_offset * config.layer_sizes[0];
        for (int l=0;l<num_layers;l++){
            int ni=config.layer_sizes[l], no=config.layer_sizes[l+1];
            bool io=(l==num_layers-1);
            dim3 bk(TILE,TILE); dim3 gg(div_ceil(B,TILE),div_ceil(no,TILE));
            if(io||config.hidden_activation==ActivationType::SIGMOID)
                fused_mixed_sigmoid<<<gg,bk>>>(d_weights_fp16[l],d_a_fp16[l],d_biases[l],d_z[l],d_a_fp16[l+1],no,B,ni);
            else
                fused_mixed_relu<<<gg,bk>>>(d_weights_fp16[l],d_a_fp16[l],d_biases[l],d_z[l],d_a_fp16[l+1],no,B,ni);
        }
    }

    void backward_and_update(int batch_offset, int B) {
        float lr=config.learning_rate, inv_B=1.0f/B;
        int od=config.layer_sizes.back();
        float* d_target=d_train_labels+batch_offset*od;

        // Output delta (FP32)
        output_delta_sigmoid_fp16<<<div_ceil(od*B,BLOCK_SIZE),BLOCK_SIZE>>>(
            d_delta[num_layers-1], d_a_fp16[num_layers], d_target, od*B, inv_B);

        // Hidden deltas — use FP32 GEMM with FP32 weights for gradient stability
        for(int l=num_layers-2;l>=0;l--){
            int nt=config.layer_sizes[l+1], nn=config.layer_sizes[l+2];
            dim3 bk(TILE,TILE), gg(div_ceil(B,TILE),div_ceil(nt,TILE));
            // Use FP32 master weights for backward — more numerically stable
            tiled_gemm_tn_fp32<<<gg,bk>>>(d_weights_fp32[l+1],d_delta[l+1],d_delta[l],nt,B,nn);
            int g1=div_ceil(nt*B,BLOCK_SIZE);
            if(config.hidden_activation==ActivationType::SIGMOID)
                sigmoid_backward_fp16<<<g1,BLOCK_SIZE>>>(d_delta[l],d_a_fp16[l+1],nt*B);
            else
                relu_backward_fp32<<<g1,BLOCK_SIZE>>>(d_delta[l],d_z[l],nt*B);
        }

        // Convert all FP16 activations to FP32 for weight update
        // a[0] comes from training images (use FP32 version directly)
        {
            int id = config.layer_sizes[0];
            CUDA_CHECK(cudaMemcpy(d_a_fp32[0], d_train_images + batch_offset * id,
                                  id * B * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        for (int l = 1; l <= num_layers; l++) {
            int n = config.layer_sizes[l];
            half2float_kernel<<<div_ceil(n*B,BLOCK_SIZE),BLOCK_SIZE>>>(d_a_fp16[l], d_a_fp32[l], n*B);
        }

        // Weight update (FP32 master weights)
        for(int l=0;l<num_layers;l++){
            int ni=config.layer_sizes[l], no=config.layer_sizes[l+1];
            dim3 bk(TILE,TILE), gg(div_ceil(ni,TILE),div_ceil(no,TILE));
            // Update FP32 masters: W32 -= lr * delta[l] * a[l]^T
            tiled_gemm_nt_update_fp32<<<gg,bk>>>(d_delta[l],d_a_fp32[l],d_weights_fp32[l],no,ni,B,-lr,1.0f);
            bias_update_fp32<<<div_ceil(no,BLOCK_SIZE),BLOCK_SIZE>>>(d_biases[l],d_delta[l],lr,no,B);

            // Sync FP16 weights from FP32 masters
            float2half_kernel<<<div_ceil(no*ni,BLOCK_SIZE),BLOCK_SIZE>>>(d_weights_fp32[l],d_weights_fp16[l],no*ni);
        }
    }

    float train_batch(int batch_start, int B) {
        forward(batch_start, B);
        int od=config.layer_sizes.back();
        // Compute loss from FP16 output (convert to FP32 for loss)
        float* d_output_fp32;
        CUDA_CHECK(cudaMalloc(&d_output_fp32, od*B*sizeof(float)));
        half2float_kernel<<<div_ceil(od*B,BLOCK_SIZE),BLOCK_SIZE>>>(d_a_fp16[num_layers],d_output_fp32,od*B);
        float loss=compute_mse_loss(d_output_fp32, d_train_labels+batch_start*od, od*B, d_loss);
        cudaFree(d_output_fp32);
        backward_and_update(batch_start, B);
        return loss;
    }

    float evaluate(const MNISTData& td) {
        int id=config.layer_sizes[0], od=config.layer_sizes.back();
        int B=config.batch_size, correct=0;
        __half* dt; CUDA_CHECK(cudaMalloc(&dt, id*B*sizeof(__half)));
        float* dt_fp32; CUDA_CHECK(cudaMalloc(&dt_fp32, id*B*sizeof(float)));
        std::vector<float> ho(od*B);

        for(int s=0;s<td.num_samples;s+=B){
            int aB=std::min(B,td.num_samples-s);
            CUDA_CHECK(cudaMemcpy(dt_fp32, td.get_image(s), id*aB*sizeof(float), cudaMemcpyHostToDevice));
            float2half_kernel<<<div_ceil(id*aB,BLOCK_SIZE),BLOCK_SIZE>>>(dt_fp32, dt, id*aB);

            __half* sv=d_a_fp16[0]; d_a_fp16[0]=dt;
            for(int l=0;l<num_layers;l++){
                int ni=config.layer_sizes[l], no=config.layer_sizes[l+1];
                bool io=(l==num_layers-1);
                dim3 bk(TILE,TILE), gg(div_ceil(aB,TILE),div_ceil(no,TILE));
                if(io||config.hidden_activation==ActivationType::SIGMOID)
                    fused_mixed_sigmoid<<<gg,bk>>>(d_weights_fp16[l],d_a_fp16[l],d_biases[l],d_z[l],d_a_fp16[l+1],no,aB,ni);
                else
                    fused_mixed_relu<<<gg,bk>>>(d_weights_fp16[l],d_a_fp16[l],d_biases[l],d_z[l],d_a_fp16[l+1],no,aB,ni);
            }
            // Convert output to FP32 for argmax
            float* d_out_fp32;
            CUDA_CHECK(cudaMalloc(&d_out_fp32, od*aB*sizeof(float)));
            half2float_kernel<<<div_ceil(od*aB,BLOCK_SIZE),BLOCK_SIZE>>>(d_a_fp16[num_layers],d_out_fp32,od*aB);
            CUDA_CHECK(cudaMemcpy(ho.data(),d_out_fp32,od*aB*sizeof(float),cudaMemcpyDeviceToHost));
            cudaFree(d_out_fp32);

            for(int i=0;i<aB;i++){
                int p=0,a=0; float mp=-1e30f,ml=-1e30f;
                for(int c=0;c<od;c++){
                    if(ho[c+i*od]>mp){mp=ho[c+i*od];p=c;}
                    if(td.labels[(s+i)*od+c]>ml){ml=td.labels[(s+i)*od+c];a=c;}
                }
                if(p==a) correct++;
            }
            d_a_fp16[0]=sv;
        }
        cudaFree(dt); cudaFree(dt_fp32);
        return (float)correct/td.num_samples;
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    printf("================================================================\n");
    printf("  V5: Mixed Precision — FP16 Storage, FP32 Accumulation\n");
    printf("================================================================\n");

    MLPConfig config = parse_args(argc, argv);
    config.print();

    MNISTData train=load_mnist(config.data_dir+"/train-images-idx3-ubyte",
                                config.data_dir+"/train-labels-idx1-ubyte");
    MNISTData test=load_mnist(config.data_dir+"/t10k-images-idx3-ubyte",
                               config.data_dir+"/t10k-labels-idx1-ubyte");

    if(config.layer_sizes.front()!=train.image_size||config.layer_sizes.back()!=train.num_classes){
        fprintf(stderr,"Error: Architecture mismatch\n"); return 1;
    }

    MLP_GPU mlp(config, train);
    int nb=train.num_samples/config.batch_size;
    double tt=0;

    printf("\nTraining: %d samples, %d batches/epoch\n\n", train.num_samples, nb);
    mlp.forward(0, config.batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CPUTimer timer; CUDATimer ctimer;
    for(int ep=0;ep<config.epochs;ep++){
        timer.start(); ctimer.start();
        float loss=0;
        for(int b=0;b<nb;b++) loss+=mlp.train_batch(b*config.batch_size,config.batch_size);
        float gt=ctimer.stop(); double wt=timer.elapsed_ms(); tt+=wt;
        float acc=mlp.evaluate(test);
        printf("Epoch %2d/%d | Loss: %.6f | Accuracy: %6.2f%% | GPU: %8.1f ms | Wall: %8.1f ms\n",
               ep+1,config.epochs,loss/nb,acc*100,gt,wt);
    }

    printf("\n================================================================\n");
    printf("  Results — V5 Mixed Precision\n");
    printf("================================================================\n");
    printf("Total training time:   %10.1f ms\n", tt);
    printf("Average epoch time:    %10.1f ms\n", tt / config.epochs);
    printf("================================================================\n");
    return 0;
}
