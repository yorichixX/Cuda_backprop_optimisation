#pragma once

// ============================================================================
// V0: CPU Baseline — Sequential Single-Threaded MLP
// ============================================================================
// This is the "sequential CPU" implementation that Sierra-Canto et al. (2010)
// benchmarked their GPU code against. It is deliberately unoptimized:
//   - No SIMD vectorization
//   - No multi-threading
//   - No cache-aware tiling
//   - Simple triple-nested loops for matrix multiplication
//
// This gives us a fair baseline to show how much GPU acceleration helps.
// All matrices are stored in COLUMN-MAJOR order for consistency with cuBLAS.
// ============================================================================

#include "mlp_config.h"
#include <vector>

class MLP_CPU {
public:
    explicit MLP_CPU(const MLPConfig& config);

    // Train on one mini-batch. Returns average MSE loss for this batch.
    float train_batch(const float* batch_images, const float* batch_labels,
                      int batch_size);

    // Forward-only pass. Stores output in internal a_[num_layers_].
    void forward(const float* inputs, int batch_size);

    // Evaluate accuracy on a dataset. Returns fraction in [0, 1].
    float evaluate(const float* images, const float* labels, int num_samples);

private:
    MLPConfig config_;
    int num_layers_;  // Number of weight layers (= layer_sizes.size() - 1)

    // --------------- Network Parameters ---------------
    // weights_[l]: shape (layer_sizes[l+1] × layer_sizes[l]), column-major
    //   - Column-major means element (i, j) is at index [i + j * rows]
    //   - Row i = output neuron i, Column j = input neuron j
    std::vector<std::vector<float>> weights_;
    // biases_[l]: shape (layer_sizes[l+1])
    std::vector<std::vector<float>> biases_;

    // --------------- Intermediate Computation Buffers ---------------
    // These are resized per forward pass based on batch_size.
    //
    // a_[l]: activations for layer l, shape (layer_sizes[l] × batch_size)
    //   - a_[0] is the input data (just a pointer copy, not owned)
    //   - a_[num_layers_] is the network output
    std::vector<std::vector<float>> a_;
    // z_[l]: pre-activation values, shape (layer_sizes[l+1] × batch_size)
    //   - z_[l] = W[l] * a_[l] + b[l]
    //   - Stored for backward pass (needed to compute activation derivatives)
    std::vector<std::vector<float>> z_;
    // delta_[l]: error gradient dL/dz_[l], same shape as z_[l]
    std::vector<std::vector<float>> delta_;

    // --------------- Internal Methods ---------------
    void backward(const float* targets, int batch_size);
    void update_weights(int batch_size);

    // Activation function dispatch (hidden vs output layer)
    float activate(float x, bool is_output_layer) const;
    float activate_deriv_from_output(float a, bool is_output_layer) const;
    float activate_deriv_from_z(float z, bool is_output_layer) const;

    // --------------- Matrix Operations (Column-Major) ---------------
    // These are deliberately naive O(M*N*K) implementations.
    // No BLAS, no SIMD — just triple-nested loops.

    // C(M×N) = A(M×K) * B(K×N)
    static void matmul_nn(const float* A, const float* B, float* C,
                          int M, int N, int K);
    // C(M×N) = A^T(M×K) * B(K×N), where A is stored as (K×M)
    static void matmul_tn(const float* A, const float* B, float* C,
                          int M, int N, int K);
    // C(M×N) = A(M×K) * B^T(K×N), where B is stored as (N×K)
    static void matmul_nt(const float* A, const float* B, float* C,
                          int M, int N, int K);
};
