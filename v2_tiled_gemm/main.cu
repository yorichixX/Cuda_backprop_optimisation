// ============================================================================
// V2: Custom Tiled GEMM with Shared Memory
// ============================================================================
//
// KEY IMPROVEMENT OVER V1:
//   Replace cuBLAS cublasSgemm() with our own hand-written tiled matrix
//   multiply kernel that uses shared memory (on-chip SRAM, ~128KB per SM).
//
// WHY THIS MATTERS:
//   In a naive GEMM, each thread loads elements from global memory (HBM).
//   For C = A * B where A is (M×K) and B is (K×N):
//     - Each element of C needs K multiply-adds
//     - Each thread loads K elements from A and K from B = 2K global loads
//     - Total loads: M*N*2K → enormous global memory traffic
//
//   With tiling:
//     - We partition A, B, C into TILE×TILE sub-blocks
//     - A tile is loaded ONCE into shared memory by TILE² threads
//     - Then reused TILE times for computation
//     - Global loads reduced by factor of TILE (= 32 here)
//
// SHARED MEMORY vs GLOBAL MEMORY:
//   - Global memory (GDDR6/HBM): ~300 GB/s bandwidth, ~400 cycle latency
//   - Shared memory (SRAM):      ~12 TB/s bandwidth, <10 cycle latency
//   - By loading tiles into shared memory, we pay the global memory cost
//     once and do all the reuse at shared memory speed.
//
// BANK CONFLICT AVOIDANCE:
//   Shared memory is organized into 32 banks (one per warp lane).
//   If two threads in the same warp access the same bank, they serialize.
//   For a 32×32 tile, column access causes bank conflicts because
//   element (row, col) maps to bank (row + col*32) % 32 = row (always same bank!).
//   FIX: Pad the shared memory to 32×33. Now bank = (row + col*33) % 32,
//   which distributes accesses across different banks.
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

// Tile size for shared memory tiling.
// 32×32 = 1024 threads per block = full occupancy on most GPUs.
// Each tile uses 32*33*4 = 4224 bytes of shared memory per input matrix.
// Two tiles (A and B): 8448 bytes — well within the 48-100KB shared mem limit.
#define TILE 32

// ============================================================================
// Tiled GEMM Kernels
// ============================================================================

// ---------------------------------------------------------------------------
// C(M×N) = A(M×K) * B(K×N) — Both non-transposed, column-major
// ---------------------------------------------------------------------------
__global__ void tiled_gemm_nn(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    // Each thread block computes one TILE×TILE sub-block of C.
    // threadIdx.y = row within tile, threadIdx.x = column within tile.

    // +1 padding eliminates bank conflicts (see header comment)
    __shared__ float tileA[TILE][TILE + 1];
    __shared__ float tileB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;  // global row of C
    int col = blockIdx.x * TILE + threadIdx.x;  // global col of C

    float sum = 0.0f;

    // Iterate over tiles along the K dimension
    int num_tiles = div_ceil(K, TILE);
    for (int t = 0; t < num_tiles; t++) {
        // --- Cooperative tile loading ---
        // All TILE² threads in the block load one element each.
        // This makes the load fully parallel and coalesced.

        // Load A tile: rows [blockIdx.y*TILE .. +TILE), cols [t*TILE .. +TILE)
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K)
            tileA[threadIdx.y][threadIdx.x] = A[row + a_col * M];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile: rows [t*TILE .. +TILE), cols [blockIdx.x*TILE .. +TILE)
        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[b_row + col * K];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // --- Barrier: ensure all threads finished loading ---
        // Without this, some threads would start computing with
        // partially-loaded tiles → wrong results.
        __syncthreads();

        // --- Compute partial dot product using shared memory ---
        // Each thread accumulates TILE multiply-adds.
        // All data comes from shared memory = fast!
        for (int i = 0; i < TILE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        // --- Barrier before next tile load ---
        // Prevents next iteration's loads from overwriting data that
        // other threads are still reading.
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row + col * M] = sum;
    }
}

// ---------------------------------------------------------------------------
// C(M×N) = A^T * B, where A stored as (K×M), column-major
// ---------------------------------------------------------------------------
// Used in backprop: dA_prev = W^T * delta
// A^T[i,p] = A[p,i] = A[p + i*K] (A is stored as K×M, lda=K)
__global__ void tiled_gemm_tn(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float tileA[TILE][TILE + 1];
    __shared__ float tileB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = div_ceil(K, TILE);
    for (int t = 0; t < num_tiles; t++) {
        // Load A^T tile: A^T[row, t*TILE+tx] = A[t*TILE+tx, row] = A[(t*TILE+tx) + row*K]
        int a_k = t * TILE + threadIdx.x;
        if (row < M && a_k < K)
            tileA[threadIdx.y][threadIdx.x] = A[a_k + row * K];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[b_row + col * K];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row + col * M] = sum;
    }
}

// ---------------------------------------------------------------------------
// C(M×N) = A * B^T, where B stored as (N×K), column-major
// ---------------------------------------------------------------------------
// Used in backprop: dW = delta * A_prev^T
// B^T[p,j] = B[j,p] = B[j + p*N] (B is stored as N×K, ldb=N)
__global__ void tiled_gemm_nt(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float tileA[TILE][TILE + 1];
    __shared__ float tileB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = div_ceil(K, TILE);
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K)
            tileA[threadIdx.y][threadIdx.x] = A[row + a_col * M];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B^T tile: B^T[t*TILE+ty, col] = B[col, t*TILE+ty] = B[col + (t*TILE+ty)*N]
        int b_k = t * TILE + threadIdx.y;
        if (col < N && b_k < K)
            tileB[threadIdx.y][threadIdx.x] = B[col + b_k * N];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row + col * M] = sum;
    }
}

// ---------------------------------------------------------------------------
// C = alpha * A_op * B_op + beta * C  (version using tiled GEMM NN + update)
// ---------------------------------------------------------------------------
// Used for in-place weight update: W = 1*W + (-lr) * delta * a^T
__global__ void tiled_gemm_nt_update(const float* A, const float* B, float* C,
                                      int M, int N, int K,
                                      float alpha, float beta_val) {
    __shared__ float tileA[TILE][TILE + 1];
    __shared__ float tileB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = div_ceil(K, TILE);
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K)
            tileA[threadIdx.y][threadIdx.x] = A[row + a_col * M];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        int b_k = t * TILE + threadIdx.y;
        if (col < N && b_k < K)
            tileB[threadIdx.y][threadIdx.x] = B[col + b_k * N];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = row + col * M;
        C[idx] = beta_val * C[idx] + alpha * sum;
    }
}

// ============================================================================
// Element-wise kernels (same as V1 — shared across versions)
// ============================================================================
__global__ void sigmoid_forward_kernel(const float* z, float* a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = fmaxf(-30.0f, fminf(30.0f, z[idx]));
        a[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void relu_forward_kernel(const float* z, float* a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { a[idx] = fmaxf(0.0f, z[idx]); }
}

__global__ void add_bias_kernel(float* Z, const float* bias, int M, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * B) { Z[idx] += bias[idx % M]; }
}

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
    if (idx < N) {
        float ai = a[idx];
        delta[idx] *= ai * (1.0f - ai);
    }
}

__global__ void relu_backward_kernel(float* delta, const float* z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { delta[idx] *= (z[idx] > 0.0f) ? 1.0f : 0.0f; }
}

__global__ void bias_update_kernel(float* bias, const float* delta,
                                   float lr, int M, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float grad = 0.0f;
        for (int j = 0; j < B; j++) grad += delta[i + j * M];
        bias[i] -= lr * grad;
    }
}

// ============================================================================
// GPU MLP Structure (same layout as V1, different GEMM calls)
// ============================================================================

struct MLP_GPU {
    MLPConfig config;
    int num_layers;

    std::vector<float*> d_weights, d_biases, d_a, d_z, d_delta;
    float *d_train_images, *d_train_labels, *d_loss;

    MLP_GPU(const MLPConfig& cfg, const MNISTData& train_data) : config(cfg) {
        num_layers = config.num_weight_layers();
        int B = config.batch_size;

        std::mt19937 rng(config.seed);
        d_weights.resize(num_layers);
        d_biases.resize(num_layers);
        d_z.resize(num_layers);
        d_delta.resize(num_layers);
        d_a.resize(num_layers + 1);

        for (int l = 0; l < num_layers; l++) {
            int n_in = config.layer_sizes[l], n_out = config.layer_sizes[l+1];
            float limit = std::sqrt(6.0f / (n_in + n_out));
            std::uniform_real_distribution<float> dist(-limit, limit);
            std::vector<float> h_w(n_out * n_in);
            for (auto& w : h_w) w = dist(rng);
            CUDA_CHECK(cudaMalloc(&d_weights[l], n_out * n_in * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights[l], h_w.data(), n_out * n_in * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_biases[l], n_out * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_biases[l], 0, n_out * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_z[l], n_out * B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_delta[l], n_out * B * sizeof(float)));
        }

        d_a[0] = nullptr;
        for (int l = 1; l <= num_layers; l++) {
            CUDA_CHECK(cudaMalloc(&d_a[l], config.layer_sizes[l] * B * sizeof(float)));
        }

        int ns = train_data.num_samples;
        CUDA_CHECK(cudaMalloc(&d_train_images, ns * config.layer_sizes[0] * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_train_labels, ns * config.layer_sizes.back() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_train_images, train_data.images.data(), ns * config.layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_train_labels, train_data.labels.data(), ns * config.layer_sizes.back() * sizeof(float), cudaMemcpyHostToDevice));
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

    // ---- Forward using tiled GEMM ----
    void forward(int batch_offset, int B) {
        d_a[0] = d_train_images + batch_offset * config.layer_sizes[0];

        for (int l = 0; l < num_layers; l++) {
            int n_in = config.layer_sizes[l], n_out = config.layer_sizes[l+1];
            int total = n_out * B;
            bool is_out = (l == num_layers - 1);

            // Z = W * A  using our tiled GEMM
            dim3 block(TILE, TILE);
            dim3 grid_gemm(div_ceil(B, TILE), div_ceil(n_out, TILE));
            tiled_gemm_nn<<<grid_gemm, block>>>(d_weights[l], d_a[l], d_z[l],
                                                  n_out, B, n_in);

            int grid1d = div_ceil(total, BLOCK_SIZE);
            add_bias_kernel<<<grid1d, BLOCK_SIZE>>>(d_z[l], d_biases[l], n_out, B);

            if (is_out || config.hidden_activation == ActivationType::SIGMOID)
                sigmoid_forward_kernel<<<grid1d, BLOCK_SIZE>>>(d_z[l], d_a[l+1], total);
            else
                relu_forward_kernel<<<grid1d, BLOCK_SIZE>>>(d_z[l], d_a[l+1], total);
        }
    }

    // ---- Backward + Update using tiled GEMM ----
    void backward_and_update(int batch_offset, int B) {
        float lr = config.learning_rate;
        float inv_B = 1.0f / B;
        int output_dim = config.layer_sizes.back();
        float* d_target = d_train_labels + batch_offset * output_dim;

        // Output delta
        int total_out = output_dim * B;
        output_delta_sigmoid_kernel<<<div_ceil(total_out, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_delta[num_layers-1], d_a[num_layers], d_target, total_out, inv_B);

        // Hidden deltas
        for (int l = num_layers - 2; l >= 0; l--) {
            int n_this = config.layer_sizes[l+1], n_next = config.layer_sizes[l+2];
            int total = n_this * B;

            // delta[l] = W[l+1]^T * delta[l+1]
            dim3 block(TILE, TILE);
            dim3 grid_gemm(div_ceil(B, TILE), div_ceil(n_this, TILE));
            tiled_gemm_tn<<<grid_gemm, block>>>(d_weights[l+1], d_delta[l+1],
                                                  d_delta[l], n_this, B, n_next);

            int grid1d = div_ceil(total, BLOCK_SIZE);
            if (config.hidden_activation == ActivationType::SIGMOID)
                sigmoid_backward_kernel<<<grid1d, BLOCK_SIZE>>>(d_delta[l], d_a[l+1], total);
            else
                relu_backward_kernel<<<grid1d, BLOCK_SIZE>>>(d_delta[l], d_z[l], total);
        }

        // Weight updates
        for (int l = 0; l < num_layers; l++) {
            int n_in = config.layer_sizes[l], n_out = config.layer_sizes[l+1];

            // W -= lr * delta * a^T
            dim3 block(TILE, TILE);
            dim3 grid_gemm(div_ceil(n_in, TILE), div_ceil(n_out, TILE));
            tiled_gemm_nt_update<<<grid_gemm, block>>>(
                d_delta[l], d_a[l], d_weights[l],
                n_out, n_in, B, -lr, 1.0f);

            bias_update_kernel<<<div_ceil(n_out, BLOCK_SIZE), BLOCK_SIZE>>>(
                d_biases[l], d_delta[l], lr, n_out, B);
        }
    }

    float train_batch(int batch_start, int B) {
        forward(batch_start, B);
        int od = config.layer_sizes.back();
        float loss = compute_mse_loss(d_a[num_layers],
                                       d_train_labels + batch_start * od,
                                       od * B, d_loss);
        backward_and_update(batch_start, B);
        return loss;
    }

    float evaluate(const MNISTData& test_data) {
        int input_dim = config.layer_sizes[0], output_dim = config.layer_sizes.back();
        int B = config.batch_size, correct = 0;
        float* d_test;
        CUDA_CHECK(cudaMalloc(&d_test, input_dim * B * sizeof(float)));
        std::vector<float> h_out(output_dim * B);

        for (int start = 0; start < test_data.num_samples; start += B) {
            int aB = std::min(B, test_data.num_samples - start);
            CUDA_CHECK(cudaMemcpy(d_test, test_data.get_image(start), input_dim * aB * sizeof(float), cudaMemcpyHostToDevice));

            float* saved = d_a[0];
            d_a[0] = d_test;
            for (int l = 0; l < num_layers; l++) {
                int ni = config.layer_sizes[l], no = config.layer_sizes[l+1];
                int total = no * aB;
                dim3 block(TILE, TILE);
                dim3 gg(div_ceil(aB, TILE), div_ceil(no, TILE));
                tiled_gemm_nn<<<gg, block>>>(d_weights[l], d_a[l], d_z[l], no, aB, ni);
                int g1 = div_ceil(total, BLOCK_SIZE);
                add_bias_kernel<<<g1, BLOCK_SIZE>>>(d_z[l], d_biases[l], no, aB);
                bool is_out = (l == num_layers - 1);
                if (is_out || config.hidden_activation == ActivationType::SIGMOID)
                    sigmoid_forward_kernel<<<g1, BLOCK_SIZE>>>(d_z[l], d_a[l+1], total);
                else
                    relu_forward_kernel<<<g1, BLOCK_SIZE>>>(d_z[l], d_a[l+1], total);
            }

            CUDA_CHECK(cudaMemcpy(h_out.data(), d_a[num_layers], output_dim * aB * sizeof(float), cudaMemcpyDeviceToHost));
            for (int s = 0; s < aB; s++) {
                int pred = 0, actual = 0;
                float mp = -1e30f, ml = -1e30f;
                for (int c = 0; c < output_dim; c++) {
                    float p = h_out[c + s * output_dim];
                    float lb = test_data.labels[(start+s)*output_dim+c];
                    if (p > mp) { mp = p; pred = c; }
                    if (lb > ml) { ml = lb; actual = c; }
                }
                if (pred == actual) correct++;
            }
            d_a[0] = saved;
        }
        cudaFree(d_test);
        return (float)correct / test_data.num_samples;
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    printf("================================================================\n");
    printf("  V2: Custom Tiled GEMM with Shared Memory\n");
    printf("================================================================\n");

    MLPConfig config = parse_args(argc, argv);
    config.print();

    MNISTData train_data = load_mnist(config.data_dir + "/train-images-idx3-ubyte",
                                       config.data_dir + "/train-labels-idx1-ubyte");
    MNISTData test_data = load_mnist(config.data_dir + "/t10k-images-idx3-ubyte",
                                      config.data_dir + "/t10k-labels-idx1-ubyte");

    if (config.layer_sizes.front() != train_data.image_size ||
        config.layer_sizes.back()  != train_data.num_classes) {
        fprintf(stderr, "Error: Architecture mismatch\n"); return 1;
    }

    MLP_GPU mlp(config, train_data);
    int num_batches = train_data.num_samples / config.batch_size;
    double total_time = 0.0;

    printf("\nTraining: %d samples, %d batches/epoch\n\n", train_data.num_samples, num_batches);
    mlp.forward(0, config.batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CPUTimer timer;
    CUDATimer ctimer;

    for (int ep = 0; ep < config.epochs; ep++) {
        timer.start(); ctimer.start();
        float loss = 0;
        for (int b = 0; b < num_batches; b++)
            loss += mlp.train_batch(b * config.batch_size, config.batch_size);
        float gt = ctimer.stop();
        double wt = timer.elapsed_ms();
        total_time += wt;
        float acc = mlp.evaluate(test_data);
        printf("Epoch %2d/%d | Loss: %.6f | Accuracy: %6.2f%% | GPU: %8.1f ms | Wall: %8.1f ms\n",
               ep+1, config.epochs, loss/num_batches, acc*100, gt, wt);
    }

    printf("\n================================================================\n");
    printf("  Results — V2 Tiled GEMM\n");
    printf("================================================================\n");
    printf("Total training time:   %10.1f ms\n", total_time);
    printf("Average epoch time:    %10.1f ms\n", total_time / config.epochs);
    printf("================================================================\n");
    return 0;
}
