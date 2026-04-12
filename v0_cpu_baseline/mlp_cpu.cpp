// ============================================================================
// V0: CPU Baseline — Full Implementation
// ============================================================================
// See mlp_cpu.h for design rationale.
//
// Matrix convention (column-major):
//   Element (row i, col j) of an (M × N) matrix A is at A[i + j * M]
//   This matches Fortran/BLAS/cuBLAS convention.
//
// Forward pass for layer l:
//   z[l] = W[l] * a[l] + b[l]           (GEMM + bias broadcast)
//   a[l+1] = activation(z[l])            (element-wise nonlinearity)
//
// Backward pass:
//   delta[L-1] = (a[L] - target) ⊙ σ'(z[L-1]) / B   (output error)
//   delta[l]   = (W[l+1]^T * delta[l+1]) ⊙ f'(z[l])  (hidden error)
//   W[l] -= lr * delta[l] * a[l]^T                     (weight update)
//   b[l] -= lr * rowsum(delta[l])                       (bias update)
// ============================================================================

#include "mlp_cpu.h"
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// Constructor — Initialize weights and allocate buffers
// ============================================================================
MLP_CPU::MLP_CPU(const MLPConfig& config) : config_(config) {
    num_layers_ = config_.num_weight_layers();
    if (num_layers_ < 1) {
        throw std::runtime_error("Need at least 2 layers (input + output)");
    }

    // ---- Xavier/Glorot Initialization ----
    // W ~ U(-limit, +limit) where limit = sqrt(6 / (n_in + n_out))
    //
    // Why Xavier? It keeps the variance of activations roughly constant
    // across layers. Without it:
    //   - Too large: activations explode → sigmoid saturates → zero gradients
    //   - Too small: activations shrink → signal disappears → no learning
    //
    // The sqrt(6/(fan_in+fan_out)) formula is derived from the constraint
    // that Var(output) = Var(input) for a linear layer with uniform weights.
    std::mt19937 rng(config_.seed);

    weights_.resize(num_layers_);
    biases_.resize(num_layers_);

    for (int l = 0; l < num_layers_; l++) {
        int n_in  = config_.layer_sizes[l];
        int n_out = config_.layer_sizes[l + 1];

        float limit = std::sqrt(6.0f / (n_in + n_out));
        std::uniform_real_distribution<float> dist(-limit, limit);

        weights_[l].resize(n_out * n_in);
        for (auto& w : weights_[l]) {
            w = dist(rng);
        }

        // Biases initialized to zero (standard practice)
        biases_[l].resize(n_out, 0.0f);
    }

    // Pre-allocate intermediate buffers (will be resized in forward())
    a_.resize(num_layers_ + 1);
    z_.resize(num_layers_);
    delta_.resize(num_layers_);
}

// ============================================================================
// Activation Functions
// ============================================================================

float MLP_CPU::activate(float x, bool is_output_layer) const {
    if (is_output_layer) {
        // Output layer: always sigmoid (bounded [0,1] for MSE loss)
        // Clamp input to [-30, 30] to prevent exp() overflow
        x = std::max(-30.0f, std::min(30.0f, x));
        return 1.0f / (1.0f + std::exp(-x));
    }
    if (config_.hidden_activation == ActivationType::RELU) {
        return std::max(0.0f, x);
    }
    // Sigmoid for hidden layers
    x = std::max(-30.0f, std::min(30.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

// Derivative given the activation output a = f(z)
// For sigmoid: f'(z) = f(z) * (1 - f(z)) = a * (1 - a)
// For ReLU: f'(z) = 1 if z > 0, else 0
//   But we don't have z here, so we use: f'(z) = (a > 0) ? 1 : 0
//   This works because ReLU(z) > 0 ⟺ z > 0
float MLP_CPU::activate_deriv_from_output(float a, bool is_output_layer) const {
    if (is_output_layer) {
        return a * (1.0f - a);  // sigmoid derivative
    }
    if (config_.hidden_activation == ActivationType::RELU) {
        return (a > 0.0f) ? 1.0f : 0.0f;
    }
    return a * (1.0f - a);  // sigmoid derivative
}

float MLP_CPU::activate_deriv_from_z(float z, bool /*is_output_layer*/) const {
    if (config_.hidden_activation == ActivationType::RELU) {
        return (z > 0.0f) ? 1.0f : 0.0f;
    }
    float a = 1.0f / (1.0f + std::exp(-std::max(-30.0f, std::min(30.0f, z))));
    return a * (1.0f - a);
}

// ============================================================================
// Matrix Multiplication (Column-Major, Naive)
// ============================================================================
// These are O(M*N*K) triple-nested loops with no optimizations.
// The inner loop has stride-M access on A (column-major), which is
// cache-unfriendly. This is INTENTIONAL — we want the baseline to be slow
// to highlight the GPU speedup.

// C(M×N) = A(M×K) * B(K×N)
void MLP_CPU::matmul_nn(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    for (int j = 0; j < N; j++) {          // column of C
        for (int i = 0; i < M; i++) {      // row of C
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {  // inner dimension
                // A[i,p] = A[i + p*M],  B[p,j] = B[p + j*K]
                sum += A[i + p * M] * B[p + j * K];
            }
            C[i + j * M] = sum;
        }
    }
}

// C(M×N) = A^T * B, where A is stored as (K×M)
// A^T[i,p] = A[p,i] = A[p + i*K]  (column-major of the stored K×M matrix)
void MLP_CPU::matmul_tn(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += A[p + i * K] * B[p + j * K];
            }
            C[i + j * M] = sum;
        }
    }
}

// C(M×N) = A * B^T, where B is stored as (N×K)
// B^T[p,j] = B[j,p] = B[j + p*N]
void MLP_CPU::matmul_nt(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += A[i + p * M] * B[j + p * N];
            }
            C[i + j * M] = sum;
        }
    }
}

// ============================================================================
// Forward Pass
// ============================================================================
void MLP_CPU::forward(const float* inputs, int batch_size) {
    int B = batch_size;

    // a_[0] = input data (just copy the pointer's data, not compute)
    int input_dim = config_.layer_sizes[0];
    a_[0].assign(inputs, inputs + input_dim * B);

    for (int l = 0; l < num_layers_; l++) {
        int n_in  = config_.layer_sizes[l];
        int n_out = config_.layer_sizes[l + 1];
        int total = n_out * B;
        bool is_output = (l == num_layers_ - 1);

        // Resize buffers for this batch size
        z_[l].resize(total);
        a_[l + 1].resize(total);

        // z[l] = W[l] * a[l]
        matmul_nn(weights_[l].data(), a_[l].data(), z_[l].data(),
                  n_out, B, n_in);

        // z[l] += bias[l]  (broadcast bias across all B samples)
        // In column-major (n_out × B), bias[i] is added to every column
        for (int j = 0; j < B; j++) {
            for (int i = 0; i < n_out; i++) {
                z_[l][i + j * n_out] += biases_[l][i];
            }
        }

        // a[l+1] = activation(z[l])
        for (int i = 0; i < total; i++) {
            a_[l + 1][i] = activate(z_[l][i], is_output);
        }
    }
}

// ============================================================================
// Backward Pass
// ============================================================================
void MLP_CPU::backward(const float* targets, int batch_size) {
    int B = batch_size;
    int L = num_layers_;

    // --------------- Output Layer Delta ---------------
    // For MSE loss with sigmoid output:
    //   L = (1/2B) * ||a[L] - target||^2
    //   dL/da[L] = (a[L] - target) / B
    //   delta[L-1] = dL/da[L] ⊙ sigmoid'(z[L-1])
    //              = (a[L] - target) / B ⊙ a[L] ⊙ (1 - a[L])
    //
    // The 1/B factor normalizes the gradient by batch size, making
    // the effective learning rate independent of batch size.
    {
        int n_out = config_.layer_sizes[L];
        int total = n_out * B;
        delta_[L - 1].resize(total);

        for (int i = 0; i < total; i++) {
            float a = a_[L][i];
            float t = targets[i];
            // Combine error and derivative in one step
            delta_[L - 1][i] = (a - t) * a * (1.0f - a) / B;
        }
    }

    // --------------- Hidden Layer Deltas ---------------
    // delta[l] = (W[l+1]^T * delta[l+1]) ⊙ activation'(z[l])
    //
    // This is the chain rule in action:
    //   dL/dz[l] = dL/da[l+1] * da[l+1]/dz[l]
    // where dL/da[l+1] = W[l+1]^T * delta[l+1] (backprop through the GEMM)
    // and   da[l+1]/dz[l] = activation'(z[l])    (backprop through nonlinearity)
    for (int l = L - 2; l >= 0; l--) {
        int n_this = config_.layer_sizes[l + 1];  // neurons in layer l+1
        int n_next = config_.layer_sizes[l + 2];  // neurons in layer l+2
        int total = n_this * B;
        bool is_output = false;  // hidden layers only

        delta_[l].resize(total);

        // delta[l] = W[l+1]^T * delta[l+1]
        // W[l+1] is (n_next × n_this), so W^T is (n_this × n_next)
        // delta[l+1] is (n_next × B)
        // Result: (n_this × B) — fits in delta_[l]
        matmul_tn(weights_[l + 1].data(), delta_[l + 1].data(),
                  delta_[l].data(), n_this, B, n_next);

        // Element-wise multiply by activation derivative
        for (int i = 0; i < total; i++) {
            float deriv;
            if (config_.hidden_activation == ActivationType::RELU) {
                deriv = (z_[l][i] > 0.0f) ? 1.0f : 0.0f;
            } else {
                float a = a_[l + 1][i];
                deriv = a * (1.0f - a);
            }
            delta_[l][i] *= deriv;
        }
    }
}

// ============================================================================
// Weight Update (SGD)
// ============================================================================
void MLP_CPU::update_weights(int batch_size) {
    int B = batch_size;
    float lr = config_.learning_rate;

    for (int l = 0; l < num_layers_; l++) {
        int n_in  = config_.layer_sizes[l];
        int n_out = config_.layer_sizes[l + 1];

        // --------------- Update Weights ---------------
        // dW = delta[l] * a[l]^T
        // W -= lr * dW
        //
        // Combined: W[i + j*n_out] -= lr * sum_b(delta[l][i + b*n_out] * a[l][j + b*n_in])
        // This avoids allocating a separate dW buffer.
        for (int i = 0; i < n_out; i++) {
            for (int j = 0; j < n_in; j++) {
                float grad = 0.0f;
                for (int b = 0; b < B; b++) {
                    grad += delta_[l][i + b * n_out] * a_[l][j + b * n_in];
                }
                weights_[l][i + j * n_out] -= lr * grad;
            }
        }

        // --------------- Update Biases ---------------
        // db = rowsum(delta[l])  (sum over batch dimension)
        // b -= lr * db
        for (int i = 0; i < n_out; i++) {
            float grad = 0.0f;
            for (int b = 0; b < B; b++) {
                grad += delta_[l][i + b * n_out];
            }
            biases_[l][i] -= lr * grad;
        }
    }
}

// ============================================================================
// Train Batch — Full forward-backward-update cycle
// ============================================================================
float MLP_CPU::train_batch(const float* batch_images, const float* batch_labels,
                           int batch_size) {
    forward(batch_images, batch_size);

    // Compute MSE loss: L = (1/2N) * sum((output - target)^2)
    int output_dim = config_.layer_sizes.back();
    int N = output_dim * batch_size;
    float loss = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = a_[num_layers_][i] - batch_labels[i];
        loss += diff * diff;
    }
    loss /= (2.0f * N);

    backward(batch_labels, batch_size);
    update_weights(batch_size);

    return loss;
}

// ============================================================================
// Evaluate — Compute classification accuracy
// ============================================================================
float MLP_CPU::evaluate(const float* images, const float* labels,
                        int num_samples) {
    int correct = 0;
    int output_dim = config_.layer_sizes.back();
    int batch = config_.batch_size;

    for (int start = 0; start < num_samples; start += batch) {
        int B = std::min(batch, num_samples - start);

        forward(images + start * config_.layer_sizes[0], B);

        // For each sample, find argmax of output and compare with label
        for (int s = 0; s < B; s++) {
            int pred = 0, actual = 0;
            float max_pred = -1e30f, max_label = -1e30f;

            for (int c = 0; c < output_dim; c++) {
                float p = a_[num_layers_][c + s * output_dim];
                float l = labels[(start + s) * output_dim + c];
                if (p > max_pred)  { max_pred  = p; pred   = c; }
                if (l > max_label) { max_label = l; actual = c; }
            }
            if (pred == actual) correct++;
        }
    }

    return static_cast<float>(correct) / num_samples;
}
