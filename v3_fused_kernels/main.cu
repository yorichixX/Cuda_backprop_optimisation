// ============================================================================
// V3: Fused Kernels — GEMM + Bias + Activation in One Launch
// ============================================================================
//
// KEY IMPROVEMENT OVER V2:
//   In V2, each layer requires 3 kernel launches:
//     1. tiled_gemm_nn → writes Z to global memory
//     2. add_bias      → reads Z, writes Z+b to global memory
//     3. sigmoid       → reads Z+b, writes σ(Z+b) to global memory
//
//   Total: 3 global memory writes + 3 global memory reads per layer.
//
//   In V3, we FUSE all three into ONE kernel:
//     1. fused_gemm_bias_act → writes ONLY final A=σ(Z+b) to global memory
//
//   Total: 1 global memory write + the GEMM loads.
//
// WHY FUSION MATTERS:
//   Global memory bandwidth is the #1 bottleneck for these kernels.
//   Each unnecessary read/write wastes ~400 cycles of memory latency.
//   For a layer with n_out=512, batch=128:
//     - Each intermediate buffer is 512*128*4 = 256KB
//     - V2 writes 3 × 256KB = 768KB per layer
//     - V3 writes 1 × 256KB = 256KB per layer (+ Z for backprop)
//     - 3x reduction in global memory traffic for element-wise ops
//
//   On the RTX 4060 with 256 GB/s bandwidth, eliminating 512KB of
//   unnecessary transfers saves ~2µs per layer — adds up over thousands
//   of batches and multiple layers.
//
// TRADEOFF:
//   We still need to store Z separately (for backprop's activation derivative).
//   So the fused kernel writes BOTH Z and A. But it does this in one pass,
//   which is still much better than 3 separate kernel launches because:
//     1. Kernel launch overhead eliminated (each launch costs ~5-10µs)
//     2. The Z values are computed in registers and written once
//     3. No need to read Z back from global memory for bias/activation
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include "data_loader.h"
#include "mlp_config.h"
#include "timer.h"
#include "cuda_utils.cuh"

#define TILE 32

// ============================================================================
// Fused Forward Kernels: GEMM + Bias + Activation in one launch
// ============================================================================

// ---------------------------------------------------------------------------
// Fused: Z = W * X + b,  A = sigmoid(Z)
// Writes BOTH Z (for backprop) and A (for next layer) to global memory.
// ---------------------------------------------------------------------------
__global__ void fused_gemm_bias_sigmoid(
        const float* W, const float* X, const float* bias,
        float* Z_out, float* A_out,
        int M, int N, int K) {
    __shared__ float tileW[TILE][TILE + 1];
    __shared__ float tileX[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = div_ceil(K, TILE);
    for (int t = 0; t < num_tiles; t++) {
        int w_col = t * TILE + threadIdx.x;
        tileW[threadIdx.y][threadIdx.x] = (row < M && w_col < K) ?
            W[row + w_col * M] : 0.0f;

        int x_row = t * TILE + threadIdx.y;
        tileX[threadIdx.y][threadIdx.x] = (x_row < K && col < N) ?
            X[x_row + col * K] : 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE; i++)
            sum += tileW[threadIdx.y][i] * tileX[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = row + col * M;
        // Add bias (loaded from global memory — but only one load per element)
        float z = sum + bias[row];
        Z_out[idx] = z;
        // Apply sigmoid directly from register — no extra global memory access
        z = fmaxf(-30.0f, fminf(30.0f, z));
        A_out[idx] = 1.0f / (1.0f + expf(-z));
    }
}

// ---------------------------------------------------------------------------
// Fused: Z = W * X + b,  A = relu(Z)
// ---------------------------------------------------------------------------
__global__ void fused_gemm_bias_relu(
        const float* W, const float* X, const float* bias,
        float* Z_out, float* A_out,
        int M, int N, int K) {
    __shared__ float tileW[TILE][TILE + 1];
    __shared__ float tileX[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = div_ceil(K, TILE);
    for (int t = 0; t < num_tiles; t++) {
        int w_col = t * TILE + threadIdx.x;
        tileW[threadIdx.y][threadIdx.x] = (row < M && w_col < K) ?
            W[row + w_col * M] : 0.0f;

        int x_row = t * TILE + threadIdx.y;
        tileX[threadIdx.y][threadIdx.x] = (x_row < K && col < N) ?
            X[x_row + col * K] : 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE; i++)
            sum += tileW[threadIdx.y][i] * tileX[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = row + col * M;
        float z = sum + bias[row];
        Z_out[idx] = z;
        A_out[idx] = fmaxf(0.0f, z);
    }
}

// ============================================================================
// Backward kernels (not fused — backprop GEMM is harder to fuse
// because the activation derivative depends on the forward pass values)
// ============================================================================

__global__ void tiled_gemm_tn(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float tA[TILE][TILE+1], tB[TILE][TILE+1];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < div_ceil(K, TILE); t++) {
        int ak = t*TILE + threadIdx.x;
        tA[threadIdx.y][threadIdx.x] = (row<M && ak<K) ? A[ak + row*K] : 0.0f;
        int br = t*TILE + threadIdx.y;
        tB[threadIdx.y][threadIdx.x] = (br<K && col<N) ? B[br + col*K] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) sum += tA[threadIdx.y][i] * tB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row + col*M] = sum;
}

__global__ void tiled_gemm_nt_update(const float* A, const float* B, float* C,
                                      int M, int N, int K, float alpha, float beta_val) {
    __shared__ float tA[TILE][TILE+1], tB[TILE][TILE+1];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < div_ceil(K, TILE); t++) {
        int ac = t*TILE + threadIdx.x;
        tA[threadIdx.y][threadIdx.x] = (row<M && ac<K) ? A[row + ac*M] : 0.0f;
        int bk = t*TILE + threadIdx.y;
        tB[threadIdx.y][threadIdx.x] = (col<N && bk<K) ? B[col + bk*N] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) sum += tA[threadIdx.y][i] * tB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) {
        int idx = row + col*M;
        C[idx] = beta_val * C[idx] + alpha * sum;
    }
}

// ============================================================================
// Element-wise backward kernels
// ============================================================================

__global__ void output_delta_sigmoid_kernel(
        float* delta, const float* output, const float* target,
        int N, float inv_B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float o = output[idx];
        delta[idx] = (o - target[idx]) * o * (1.0f - o) * inv_B;
    }
}

__global__ void sigmoid_backward_kernel(float* delta, const float* a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { float ai = a[idx]; delta[idx] *= ai * (1.0f - ai); }
}

__global__ void relu_backward_kernel(float* delta, const float* z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { delta[idx] *= (z[idx] > 0.0f) ? 1.0f : 0.0f; }
}

__global__ void bias_update_kernel(float* bias, const float* delta, float lr, int M, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float g = 0.0f;
        for (int j = 0; j < B; j++) g += delta[i + j*M];
        bias[i] -= lr * g;
    }
}

// ============================================================================
// GPU MLP
// ============================================================================

struct MLP_GPU {
    MLPConfig config;
    int num_layers;
    std::vector<float*> d_weights, d_biases, d_a, d_z, d_delta;
    float *d_train_images, *d_train_labels, *d_loss;

    MLP_GPU(const MLPConfig& cfg, const MNISTData& td) : config(cfg) {
        num_layers = config.num_weight_layers();
        int B = config.batch_size;
        std::mt19937 rng(config.seed);

        d_weights.resize(num_layers); d_biases.resize(num_layers);
        d_z.resize(num_layers); d_delta.resize(num_layers);
        d_a.resize(num_layers + 1);

        for (int l = 0; l < num_layers; l++) {
            int ni = config.layer_sizes[l], no = config.layer_sizes[l+1];
            float lim = std::sqrt(6.0f / (ni + no));
            std::uniform_real_distribution<float> dist(-lim, lim);
            std::vector<float> hw(no * ni);
            for (auto& w : hw) w = dist(rng);
            CUDA_CHECK(cudaMalloc(&d_weights[l], no*ni*sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights[l], hw.data(), no*ni*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_biases[l], no*sizeof(float)));
            CUDA_CHECK(cudaMemset(d_biases[l], 0, no*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_z[l], no*B*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_delta[l], no*B*sizeof(float)));
        }
        d_a[0] = nullptr;
        for (int l = 1; l <= num_layers; l++)
            CUDA_CHECK(cudaMalloc(&d_a[l], config.layer_sizes[l]*B*sizeof(float)));

        int ns = td.num_samples;
        CUDA_CHECK(cudaMalloc(&d_train_images, ns*config.layer_sizes[0]*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_train_labels, ns*config.layer_sizes.back()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_train_images, td.images.data(), ns*config.layer_sizes[0]*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_train_labels, td.labels.data(), ns*config.layer_sizes.back()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    }

    ~MLP_GPU() {
        for (int l = 0; l < num_layers; l++) {
            cudaFree(d_weights[l]); cudaFree(d_biases[l]);
            cudaFree(d_z[l]); cudaFree(d_delta[l]);
        }
        for (int l = 1; l <= num_layers; l++) cudaFree(d_a[l]);
        cudaFree(d_train_images); cudaFree(d_train_labels); cudaFree(d_loss);
    }

    // ---- Fused forward pass ----
    // ONE kernel launch per layer instead of THREE
    void forward(int batch_offset, int B) {
        d_a[0] = d_train_images + batch_offset * config.layer_sizes[0];

        for (int l = 0; l < num_layers; l++) {
            int ni = config.layer_sizes[l], no = config.layer_sizes[l+1];
            bool is_out = (l == num_layers - 1);

            dim3 block(TILE, TILE);
            dim3 grid(div_ceil(B, TILE), div_ceil(no, TILE));

            // Single fused kernel: GEMM + bias + activation
            if (is_out || config.hidden_activation == ActivationType::SIGMOID)
                fused_gemm_bias_sigmoid<<<grid, block>>>(
                    d_weights[l], d_a[l], d_biases[l],
                    d_z[l], d_a[l+1], no, B, ni);
            else
                fused_gemm_bias_relu<<<grid, block>>>(
                    d_weights[l], d_a[l], d_biases[l],
                    d_z[l], d_a[l+1], no, B, ni);
        }
    }

    void backward_and_update(int batch_offset, int B) {
        float lr = config.learning_rate, inv_B = 1.0f / B;
        int od = config.layer_sizes.back();
        float* d_target = d_train_labels + batch_offset * od;

        output_delta_sigmoid_kernel<<<div_ceil(od*B, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_delta[num_layers-1], d_a[num_layers], d_target, od*B, inv_B);

        for (int l = num_layers-2; l >= 0; l--) {
            int nt = config.layer_sizes[l+1], nn = config.layer_sizes[l+2];
            dim3 bk(TILE, TILE);
            dim3 gg(div_ceil(B, TILE), div_ceil(nt, TILE));
            tiled_gemm_tn<<<gg, bk>>>(d_weights[l+1], d_delta[l+1], d_delta[l], nt, B, nn);
            int g1 = div_ceil(nt*B, BLOCK_SIZE);
            if (config.hidden_activation == ActivationType::SIGMOID)
                sigmoid_backward_kernel<<<g1, BLOCK_SIZE>>>(d_delta[l], d_a[l+1], nt*B);
            else
                relu_backward_kernel<<<g1, BLOCK_SIZE>>>(d_delta[l], d_z[l], nt*B);
        }

        for (int l = 0; l < num_layers; l++) {
            int ni = config.layer_sizes[l], no = config.layer_sizes[l+1];
            dim3 bk(TILE, TILE);
            dim3 gg(div_ceil(ni, TILE), div_ceil(no, TILE));
            tiled_gemm_nt_update<<<gg, bk>>>(d_delta[l], d_a[l], d_weights[l], no, ni, B, -lr, 1.0f);
            bias_update_kernel<<<div_ceil(no, BLOCK_SIZE), BLOCK_SIZE>>>(d_biases[l], d_delta[l], lr, no, B);
        }
    }

    float train_batch(int batch_start, int B) {
        forward(batch_start, B);
        int od = config.layer_sizes.back();
        float loss = compute_mse_loss(d_a[num_layers], d_train_labels + batch_start*od, od*B, d_loss);
        backward_and_update(batch_start, B);
        return loss;
    }

    float evaluate(const MNISTData& td) {
        int id = config.layer_sizes[0], od = config.layer_sizes.back();
        int B = config.batch_size, correct = 0;
        float* dt; CUDA_CHECK(cudaMalloc(&dt, id*B*sizeof(float)));
        std::vector<float> ho(od*B);
        for (int s = 0; s < td.num_samples; s += B) {
            int aB = std::min(B, td.num_samples - s);
            CUDA_CHECK(cudaMemcpy(dt, td.get_image(s), id*aB*sizeof(float), cudaMemcpyHostToDevice));
            float* sv = d_a[0]; d_a[0] = dt;
            for (int l = 0; l < num_layers; l++) {
                int ni = config.layer_sizes[l], no = config.layer_sizes[l+1];
                bool io = (l == num_layers-1);
                dim3 bk(TILE, TILE); dim3 gg(div_ceil(aB, TILE), div_ceil(no, TILE));
                if (io || config.hidden_activation == ActivationType::SIGMOID)
                    fused_gemm_bias_sigmoid<<<gg, bk>>>(d_weights[l], d_a[l], d_biases[l], d_z[l], d_a[l+1], no, aB, ni);
                else
                    fused_gemm_bias_relu<<<gg, bk>>>(d_weights[l], d_a[l], d_biases[l], d_z[l], d_a[l+1], no, aB, ni);
            }
            CUDA_CHECK(cudaMemcpy(ho.data(), d_a[num_layers], od*aB*sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < aB; i++) {
                int p=0, a=0; float mp=-1e30f, ml=-1e30f;
                for (int c = 0; c < od; c++) {
                    if (ho[c+i*od]>mp) { mp=ho[c+i*od]; p=c; }
                    if (td.labels[(s+i)*od+c]>ml) { ml=td.labels[(s+i)*od+c]; a=c; }
                }
                if (p==a) correct++;
            }
            d_a[0] = sv;
        }
        cudaFree(dt);
        return (float)correct / td.num_samples;
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    printf("================================================================\n");
    printf("  V3: Fused Kernels — GEMM + Bias + Activation\n");
    printf("================================================================\n");

    MLPConfig config = parse_args(argc, argv);
    config.print();

    MNISTData train = load_mnist(config.data_dir + "/train-images-idx3-ubyte",
                                  config.data_dir + "/train-labels-idx1-ubyte");
    MNISTData test = load_mnist(config.data_dir + "/t10k-images-idx3-ubyte",
                                 config.data_dir + "/t10k-labels-idx1-ubyte");

    if (config.layer_sizes.front() != train.image_size ||
        config.layer_sizes.back()  != train.num_classes) {
        fprintf(stderr, "Error: Architecture mismatch\n"); return 1;
    }

    MLP_GPU mlp(config, train);
    int nb = train.num_samples / config.batch_size;
    double tt = 0;

    printf("\nTraining: %d samples, %d batches/epoch\n\n", train.num_samples, nb);
    mlp.forward(0, config.batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CPUTimer timer; CUDATimer ctimer;
    for (int ep = 0; ep < config.epochs; ep++) {
        timer.start(); ctimer.start();
        float loss = 0;
        for (int b = 0; b < nb; b++)
            loss += mlp.train_batch(b * config.batch_size, config.batch_size);
        float gt = ctimer.stop(); double wt = timer.elapsed_ms(); tt += wt;
        float acc = mlp.evaluate(test);
        printf("Epoch %2d/%d | Loss: %.6f | Accuracy: %6.2f%% | GPU: %8.1f ms | Wall: %8.1f ms\n",
               ep+1, config.epochs, loss/nb, acc*100, gt, wt);
    }

    printf("\n================================================================\n");
    printf("  Results — V3 Fused Kernels\n");
    printf("================================================================\n");
    printf("Total training time:   %10.1f ms\n", tt);
    printf("Average epoch time:    %10.1f ms\n", tt / config.epochs);
    printf("================================================================\n");
    return 0;
}
