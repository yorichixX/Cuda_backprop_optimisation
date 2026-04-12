// ============================================================================
// V1: Naive CUDA — cuBLAS + Custom Kernels (The Paper's Approach)
// ============================================================================
//
// This version replicates the methodology of Sierra-Canto et al. (2010):
//   - cuBLAS cublasSgemm() for all matrix multiplications (GEMM)
//   - Hand-written CUDA kernels for element-wise operations:
//     activation functions, bias addition, weight updates, error computation
//
// Key characteristics of this "naive" GPU approach:
//   1. SEPARATE kernel launches for each operation (GEMM, bias, activation)
//      → Each writes to global memory, next reads it back = wasted bandwidth
//   2. No shared memory optimization in custom kernels
//      → Every element-wise kernel does one global memory read + one write
//   3. All training data loaded to GPU at startup
//      → Simple but assumes dataset fits in GPU memory
//
// This version is "naive" compared to our later optimizations, but it already
// demonstrates massive speedup over CPU because:
//   - cuBLAS GEMM is extremely well-optimized (uses tensor cores internally)
//   - 3072 CUDA cores running in parallel vs 1 CPU thread
//   - GPU memory bandwidth is 10-20x higher than CPU DDR
//
// The purpose of this version is to establish a GPU baseline that matches
// the paper's approach, then show how V2-V6 improve upon it.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cublas_v2.h>

#include "data_loader.h"
#include "mlp_config.h"
#include "timer.h"
#include "cuda_utils.cuh"

// ============================================================================
// CUDA Kernels — Element-wise operations
// ============================================================================
// Each kernel follows the same pattern:
//   1. Compute global thread index
//   2. Bounds check (idx < N)
//   3. Perform one element-wise operation
//   4. Write result to global memory
//
// Thread block size is 256 (BLOCK_SIZE from cuda_utils.cuh).
// Grid size = ceil(N / 256).
//
// These are intentionally simple 1D kernels. No shared memory, no tiling.
// The parallelism comes from launching thousands of threads, one per element.
// ============================================================================

// ---------------------------------------------------------------------------
// Forward pass: apply activation function to pre-activation values
// ---------------------------------------------------------------------------
// Reads z[i], writes a[i] = activation(z[i])
// Separate input/output buffers because we need z for the backward pass.
__global__ void sigmoid_forward_kernel(const float* z, float* a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Clamp to prevent exp() overflow for very negative inputs
        float val = fmaxf(-30.0f, fminf(30.0f, z[idx]));
        a[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void relu_forward_kernel(const float* z, float* a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = fmaxf(0.0f, z[idx]);
    }
}

// ---------------------------------------------------------------------------
// Bias addition: Z[i,j] += bias[i] for all j in [0, batch_size)
// ---------------------------------------------------------------------------
// Z is (M × B) column-major. bias is (M × 1).
// Each element Z[i + j*M] gets bias[i] added.
// We use modular indexing: row = idx % M gives the bias index.
__global__ void add_bias_kernel(float* Z, const float* bias, int M, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * B;
    if (idx < total) {
        int row = idx % M;
        Z[idx] += bias[row];
    }
}

// ---------------------------------------------------------------------------
// Output layer delta for sigmoid + MSE loss
// ---------------------------------------------------------------------------
// delta = (output - target) * sigmoid'(z) / batch_size
//       = (output - target) * output * (1 - output) / batch_size
//
// The 1/B factor normalizes the gradient so the effective learning rate
// doesn't depend on batch size.
__global__ void output_delta_sigmoid_kernel(
        float* delta, const float* output, const float* target,
        int N, float inv_batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float o = output[idx];
        float t = target[idx];
        delta[idx] = (o - t) * o * (1.0f - o) * inv_batch_size;
    }
}

// ---------------------------------------------------------------------------
// Hidden layer backward: multiply delta by activation derivative
// ---------------------------------------------------------------------------
// For sigmoid: delta[i] *= a[i] * (1 - a[i])
// For ReLU:    delta[i] *= (z[i] > 0) ? 1 : 0
//
// delta already contains W[l+1]^T * delta[l+1] (computed by cuBLAS GEMM).
// This kernel applies the activation derivative element-wise.
__global__ void sigmoid_backward_kernel(float* delta, const float* a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float ai = a[idx];
        delta[idx] *= ai * (1.0f - ai);
    }
}

__global__ void relu_backward_kernel(float* delta, const float* z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        delta[idx] *= (z[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Bias gradient computation + SGD update
// ---------------------------------------------------------------------------
// db[i] = sum_j(delta[i + j * M])   (sum over batch dimension)
// bias[i] -= lr * db[i]
//
// Each thread handles one bias element (one row of the delta matrix).
// The inner loop sums over columns (batch samples).
// This has non-coalesced memory access (stride M between batch elements),
// but it's a small kernel and not performance-critical.
__global__ void bias_update_kernel(float* bias, const float* delta,
                                   float lr, int M, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float grad = 0.0f;
        for (int j = 0; j < B; j++) {
            grad += delta[i + j * M];
        }
        bias[i] -= lr * grad;
    }
}

// ============================================================================
// GPU MLP Structure
// ============================================================================

struct MLP_GPU {
    MLPConfig config;
    int num_layers;
    cublasHandle_t cublas;

    // Device memory for network parameters
    std::vector<float*> d_weights;   // W[l]: (n_out × n_in) column-major
    std::vector<float*> d_biases;    // b[l]: (n_out)

    // Device memory for intermediate computations
    std::vector<float*> d_a;         // a[l]: activations. a[0] points into d_train_images
    std::vector<float*> d_z;         // z[l]: pre-activation = W*a + b
    std::vector<float*> d_delta;     // delta[l]: error gradients

    // Device memory for training data (loaded once at startup)
    float* d_train_images = nullptr;
    float* d_train_labels = nullptr;

    // Scalar for loss reduction
    float* d_loss = nullptr;

    // ----------------------------------------------------------------
    // Constructor: allocate GPU memory, initialize weights
    // ----------------------------------------------------------------
    MLP_GPU(const MLPConfig& cfg, const MNISTData& train_data) : config(cfg) {
        num_layers = config.num_weight_layers();

        // Create cuBLAS handle
        CUBLAS_CHECK(cublasCreate(&cublas));

        // ---- Allocate parameter memory ----
        d_weights.resize(num_layers);
        d_biases.resize(num_layers);
        d_z.resize(num_layers);
        d_delta.resize(num_layers);
        d_a.resize(num_layers + 1);

        // Xavier initialization on CPU, then copy to GPU
        std::mt19937 rng(config.seed);
        for (int l = 0; l < num_layers; l++) {
            int n_in  = config.layer_sizes[l];
            int n_out = config.layer_sizes[l + 1];

            // Xavier/Glorot uniform initialization
            float limit = std::sqrt(6.0f / (n_in + n_out));
            std::uniform_real_distribution<float> dist(-limit, limit);

            // Allocate and initialize weights
            std::vector<float> h_w(n_out * n_in);
            for (auto& w : h_w) w = dist(rng);
            CUDA_CHECK(cudaMalloc(&d_weights[l], n_out * n_in * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights[l], h_w.data(),
                                  n_out * n_in * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Allocate and zero biases
            CUDA_CHECK(cudaMalloc(&d_biases[l], n_out * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_biases[l], 0, n_out * sizeof(float)));
        }

        // ---- Allocate intermediate buffers ----
        // Size based on max batch size
        int B = config.batch_size;
        for (int l = 0; l < num_layers; l++) {
            int n_out = config.layer_sizes[l + 1];
            CUDA_CHECK(cudaMalloc(&d_z[l],     n_out * B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_delta[l],  n_out * B * sizeof(float)));
        }
        // a[0] will point into training data; allocate a[1]..a[L]
        d_a[0] = nullptr;  // set per batch
        for (int l = 1; l <= num_layers; l++) {
            int n = config.layer_sizes[l];
            CUDA_CHECK(cudaMalloc(&d_a[l], n * B * sizeof(float)));
        }

        // ---- Upload all training data to GPU ----
        // This avoids per-batch transfers. MNIST easily fits in 8GB GPU memory:
        //   60000 * 784 * 4 bytes ≈ 179 MB (images)
        //   60000 *  10 * 4 bytes ≈   2 MB (labels)
        int num_samples = train_data.num_samples;
        CUDA_CHECK(cudaMalloc(&d_train_images,
                              num_samples * config.layer_sizes[0] * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_train_labels,
                              num_samples * config.layer_sizes.back() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_train_images, train_data.images.data(),
                              num_samples * config.layer_sizes[0] * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_train_labels, train_data.labels.data(),
                              num_samples * config.layer_sizes.back() * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Loss scalar
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    }

    // ----------------------------------------------------------------
    // Destructor: free all GPU memory
    // ----------------------------------------------------------------
    ~MLP_GPU() {
        for (int l = 0; l < num_layers; l++) {
            cudaFree(d_weights[l]);
            cudaFree(d_biases[l]);
            cudaFree(d_z[l]);
            cudaFree(d_delta[l]);
        }
        for (int l = 1; l <= num_layers; l++) {
            cudaFree(d_a[l]);
        }
        cudaFree(d_train_images);
        cudaFree(d_train_labels);
        cudaFree(d_loss);
        cublasDestroy(cublas);
    }

    // ----------------------------------------------------------------
    // Forward Pass using cuBLAS
    // ----------------------------------------------------------------
    // For each layer l:
    //   1. Z[l] = W[l] * A[l]           (cuBLAS GEMM)
    //   2. Z[l] += bias[l]              (custom kernel)
    //   3. A[l+1] = activation(Z[l])    (custom kernel)
    //
    // Three separate kernel launches per layer — this is what V3 will fuse.
    // ----------------------------------------------------------------
    void forward(int batch_offset, int B) {
        // a[0] = pointer to batch images in GPU memory
        d_a[0] = d_train_images + batch_offset * config.layer_sizes[0];

        float alpha = 1.0f, beta = 0.0f;

        for (int l = 0; l < num_layers; l++) {
            int n_in  = config.layer_sizes[l];
            int n_out = config.layer_sizes[l + 1];
            int total = n_out * B;
            bool is_output = (l == num_layers - 1);

            // ----- Step 1: GEMM via cuBLAS -----
            // Z[l] = W[l] * A[l]
            // cublasSgemm(handle, opA, opB, m, n, k, &α, A, lda, B, ldb, &β, C, ldc)
            //   Computes: C = α * op(A) * op(B) + β * C
            //   op(A) = W[l]    (m=n_out, k=n_in)
            //   op(B) = A[l]    (k=n_in,  n=B)
            //   C      = Z[l]   (m=n_out, n=B)
            CUBLAS_CHECK(cublasSgemm(cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n_out, B, n_in,
                &alpha,
                d_weights[l], n_out,     // W: (n_out × n_in), lda = n_out
                d_a[l],       n_in,      // A: (n_in × B),     ldb = n_in
                &beta,
                d_z[l],       n_out));   // Z: (n_out × B),    ldc = n_out

            // ----- Step 2: Add bias -----
            // This is a separate kernel launch = separate global memory round-trip
            int grid = div_ceil(total, BLOCK_SIZE);
            add_bias_kernel<<<grid, BLOCK_SIZE>>>(d_z[l], d_biases[l], n_out, B);

            // ----- Step 3: Activation -----
            // Another separate kernel launch
            if (is_output || config.hidden_activation == ActivationType::SIGMOID) {
                sigmoid_forward_kernel<<<grid, BLOCK_SIZE>>>(d_z[l], d_a[l+1], total);
            } else {
                relu_forward_kernel<<<grid, BLOCK_SIZE>>>(d_z[l], d_a[l+1], total);
            }
        }
    }

    // ----------------------------------------------------------------
    // Backward Pass + Weight Update using cuBLAS
    // ----------------------------------------------------------------
    void backward_and_update(int batch_offset, int B) {
        float lr = config.learning_rate;
        float inv_B = 1.0f / B;
        float alpha, beta;
        int output_dim = config.layer_sizes.back();

        // Get pointer to this batch's labels in GPU memory
        float* d_target = d_train_labels + batch_offset * output_dim;

        // ---- Output layer delta ----
        int total_out = output_dim * B;
        int grid = div_ceil(total_out, BLOCK_SIZE);
        output_delta_sigmoid_kernel<<<grid, BLOCK_SIZE>>>(
            d_delta[num_layers - 1], d_a[num_layers], d_target,
            total_out, inv_B);

        // ---- Hidden layer deltas (backpropagate) ----
        for (int l = num_layers - 2; l >= 0; l--) {
            int n_this = config.layer_sizes[l + 1];
            int n_next = config.layer_sizes[l + 2];
            int total  = n_this * B;

            // delta[l] = W[l+1]^T * delta[l+1]
            // cuBLAS: C = α * W^T * delta + β * C
            //   op(A) = W[l+1]^T (m=n_this, k=n_next)
            //   op(B) = delta[l+1] (k=n_next, n=B)
            //   C     = delta[l]   (m=n_this, n=B)
            alpha = 1.0f; beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n_this, B, n_next,
                &alpha,
                d_weights[l + 1], n_next,   // W stored as (n_next × n_this), lda = n_next
                d_delta[l + 1],   n_next,   // delta[l+1]: (n_next × B), ldb = n_next
                &beta,
                d_delta[l],       n_this)); // delta[l]: (n_this × B), ldc = n_this

            // Multiply by activation derivative
            grid = div_ceil(total, BLOCK_SIZE);
            if (config.hidden_activation == ActivationType::SIGMOID) {
                sigmoid_backward_kernel<<<grid, BLOCK_SIZE>>>(
                    d_delta[l], d_a[l + 1], total);
            } else {
                relu_backward_kernel<<<grid, BLOCK_SIZE>>>(
                    d_delta[l], d_z[l], total);
            }
        }

        // ---- Weight updates (SGD) ----
        for (int l = 0; l < num_layers; l++) {
            int n_in  = config.layer_sizes[l];
            int n_out = config.layer_sizes[l + 1];

            // W[l] -= lr * delta[l] * a[l]^T
            // cuBLAS trick: C = α * A * B^T + β * C  with α=-lr, β=1
            //   → W = 1*W + (-lr) * delta * a^T = W - lr * dW
            // This does the gradient computation AND weight update in ONE call!
            float neg_lr = -lr;
            float one = 1.0f;
            CUBLAS_CHECK(cublasSgemm(cublas,
                CUBLAS_OP_N, CUBLAS_OP_T,
                n_out, n_in, B,
                &neg_lr,
                d_delta[l], n_out,   // delta: (n_out × B)
                d_a[l],     n_in,    // a: (n_in × B), transposed to (B × n_in)
                &one,
                d_weights[l], n_out)); // W: (n_out × n_in), updated in-place!

            // Bias update
            grid = div_ceil(n_out, BLOCK_SIZE);
            bias_update_kernel<<<grid, BLOCK_SIZE>>>(
                d_biases[l], d_delta[l], lr, n_out, B);
        }
    }

    // ----------------------------------------------------------------
    // Train one batch: forward + backward + update
    // ----------------------------------------------------------------
    float train_batch(int batch_start, int B) {
        forward(batch_start, B);

        // Compute loss (for reporting only — not on the critical path)
        int output_dim = config.layer_sizes.back();
        float* d_target = d_train_labels + batch_start * output_dim;
        float loss = compute_mse_loss(d_a[num_layers], d_target,
                                       output_dim * B, d_loss);

        backward_and_update(batch_start, B);
        return loss;
    }

    // ----------------------------------------------------------------
    // Evaluate accuracy on test data
    // ----------------------------------------------------------------
    float evaluate(const MNISTData& test_data) {
        int input_dim  = config.layer_sizes[0];
        int output_dim = config.layer_sizes.back();
        int B = config.batch_size;
        int correct = 0;

        // Temporary GPU buffer for test input
        float* d_test_input;
        CUDA_CHECK(cudaMalloc(&d_test_input, input_dim * B * sizeof(float)));

        // Host buffer for output (to compute argmax on CPU)
        std::vector<float> h_output(output_dim * B);

        for (int start = 0; start < test_data.num_samples; start += B) {
            int actual_B = std::min(B, test_data.num_samples - start);

            // Upload test batch to GPU
            CUDA_CHECK(cudaMemcpy(d_test_input,
                                  test_data.get_image(start),
                                  input_dim * actual_B * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Forward pass (temporarily set a[0] to test data)
            float* saved_a0 = d_a[0];
            d_a[0] = d_test_input;

            float alpha = 1.0f, beta = 0.0f;
            for (int l = 0; l < num_layers; l++) {
                int n_in  = config.layer_sizes[l];
                int n_out = config.layer_sizes[l + 1];
                int total = n_out * actual_B;
                bool is_output = (l == num_layers - 1);

                CUBLAS_CHECK(cublasSgemm(cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n_out, actual_B, n_in,
                    &alpha, d_weights[l], n_out,
                    d_a[l], n_in,
                    &beta, d_z[l], n_out));

                int grid = div_ceil(total, BLOCK_SIZE);
                add_bias_kernel<<<grid, BLOCK_SIZE>>>(d_z[l], d_biases[l], n_out, actual_B);

                if (is_output || config.hidden_activation == ActivationType::SIGMOID) {
                    sigmoid_forward_kernel<<<grid, BLOCK_SIZE>>>(d_z[l], d_a[l+1], total);
                } else {
                    relu_forward_kernel<<<grid, BLOCK_SIZE>>>(d_z[l], d_a[l+1], total);
                }
            }

            // Copy output to CPU for argmax
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_a[num_layers],
                                  output_dim * actual_B * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // Argmax comparison
            for (int s = 0; s < actual_B; s++) {
                int pred = 0, actual = 0;
                float max_p = -1e30f, max_l = -1e30f;
                for (int c = 0; c < output_dim; c++) {
                    float p = h_output[c + s * output_dim];
                    float lbl = test_data.labels[(start + s) * output_dim + c];
                    if (p > max_p)   { max_p = p;   pred   = c; }
                    if (lbl > max_l) { max_l = lbl; actual = c; }
                }
                if (pred == actual) correct++;
            }

            d_a[0] = saved_a0;
        }

        cudaFree(d_test_input);
        return static_cast<float>(correct) / test_data.num_samples;
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    printf("================================================================\n");
    printf("  V1: Naive CUDA — cuBLAS + Custom Kernels (Paper's Approach)\n");
    printf("================================================================\n");

    MLPConfig config = parse_args(argc, argv);
    config.print();

    // Load MNIST
    printf("Loading MNIST data from '%s'...\n", config.data_dir.c_str());
    MNISTData train_data = load_mnist(
        config.data_dir + "/train-images-idx3-ubyte",
        config.data_dir + "/train-labels-idx1-ubyte");
    MNISTData test_data = load_mnist(
        config.data_dir + "/t10k-images-idx3-ubyte",
        config.data_dir + "/t10k-labels-idx1-ubyte");
    printf("\n");

    // Validate
    if (config.layer_sizes.front() != train_data.image_size ||
        config.layer_sizes.back()  != train_data.num_classes) {
        fprintf(stderr, "Error: Architecture doesn't match data dimensions\n");
        return 1;
    }

    // Create MLP and upload data to GPU
    MLP_GPU mlp(config, train_data);

    int num_batches = train_data.num_samples / config.batch_size;
    double total_train_time = 0.0;

    printf("Training: %d samples, %d batches/epoch, batch_size=%d\n\n",
           train_data.num_samples, num_batches, config.batch_size);

    // Warm-up: run one forward pass to initialize CUDA context
    // (first CUDA call has ~100ms overhead for context creation)
    mlp.forward(0, config.batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CPUTimer epoch_timer;
    CUDATimer cuda_timer;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        epoch_timer.start();
        cuda_timer.start();
        float total_loss = 0.0f;

        for (int b = 0; b < num_batches; b++) {
            total_loss += mlp.train_batch(b * config.batch_size, config.batch_size);
        }

        float gpu_time = cuda_timer.stop();
        double wall_time = epoch_timer.elapsed_ms();
        total_train_time += wall_time;

        float accuracy = mlp.evaluate(test_data);

        printf("Epoch %2d/%d | Loss: %.6f | Accuracy: %6.2f%% | "
               "GPU: %8.1f ms | Wall: %8.1f ms\n",
               epoch + 1, config.epochs,
               total_loss / num_batches,
               accuracy * 100.0f,
               gpu_time, wall_time);
    }

    printf("\n================================================================\n");
    printf("  Results — V1 Naive CUDA (cuBLAS)\n");
    printf("================================================================\n");
    printf("Total training time:   %10.1f ms\n", total_train_time);
    printf("Average epoch time:    %10.1f ms\n", total_train_time / config.epochs);
    printf("================================================================\n");

    return 0;
}
