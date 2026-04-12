#pragma once

// ============================================================================
// MLP Configuration
// ============================================================================
// Shared configuration struct used by ALL versions (CPU and GPU).
// Defines network architecture, training hyperparameters, and data paths.
// Parses command-line arguments so we can benchmark different configurations
// without recompiling.
// ============================================================================

#include <vector>
#include <string>
#include <iostream>

// ---------------------------------------------------------------------------
// Activation function selector
// ---------------------------------------------------------------------------
// SIGMOID: The classic σ(x) = 1/(1+e^{-x}). Used in the original paper.
//   - Smooth gradient, bounded output [0,1]
//   - Problem: vanishing gradients for deep networks (σ'(x) ≤ 0.25)
//
// RELU: f(x) = max(0, x). The modern default.
//   - Constant gradient for x>0, no vanishing gradient problem
//   - Sparse activations (many zeros) → faster convergence
//   - "Dying ReLU" problem: neurons can get stuck at 0
// ---------------------------------------------------------------------------
enum class ActivationType {
    SIGMOID,
    RELU
};

struct MLPConfig {
    // ---- Network Architecture ----
    // List of layer sizes, e.g., {784, 512, 256, 10}
    // First element = input dimension, last = output dimension
    // Everything in between = hidden layers
    std::vector<int> layer_sizes = {784, 128, 10};

    // ---- Training Hyperparameters ----
    float learning_rate  = 0.1f;    // SGD step size
    int   epochs         = 10;      // Full passes over training data
    int   batch_size     = 128;     // Mini-batch size for SGD
    int   seed           = 42;      // RNG seed for reproducibility

    // ---- Activation ----
    // This controls hidden layer activation. Output layer always uses sigmoid
    // (matching the paper's MSE + sigmoid output setup).
    ActivationType hidden_activation = ActivationType::SIGMOID;

    // ---- Data ----
    std::string data_dir = "data";

    // ---- Derived ----
    int num_weight_layers() const {
        return static_cast<int>(layer_sizes.size()) - 1;
    }

    void print() const {
        std::cout << "\n=== MLP Configuration ===" << std::endl;
        std::cout << "Architecture: ";
        for (size_t i = 0; i < layer_sizes.size(); i++) {
            std::cout << layer_sizes[i];
            if (i < layer_sizes.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
        std::cout << "Hidden activation: "
                  << (hidden_activation == ActivationType::SIGMOID ? "sigmoid" : "relu")
                  << std::endl;
        std::cout << "Learning rate: " << learning_rate << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "===========================\n" << std::endl;
    }
};

// ---------------------------------------------------------------------------
// Command-line argument parser
// ---------------------------------------------------------------------------
// Usage examples:
//   v0_cpu --arch 784,512,256,10 --lr 0.01 --epochs 20 --batch 256
//   v1_naive --activation relu --data ./mnist_data
// ---------------------------------------------------------------------------
inline MLPConfig parse_args(int argc, char* argv[]) {
    MLPConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        }
        else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--batch" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        }
        else if (arg == "--activation" && i + 1 < argc) {
            std::string act = argv[++i];
            config.hidden_activation =
                (act == "relu") ? ActivationType::RELU : ActivationType::SIGMOID;
        }
        else if (arg == "--arch" && i + 1 < argc) {
            // Parse comma-separated layer sizes: "784,512,256,10"
            config.layer_sizes.clear();
            std::string arch = argv[++i];
            size_t pos = 0;
            while ((pos = arch.find(',')) != std::string::npos) {
                config.layer_sizes.push_back(std::stoi(arch.substr(0, pos)));
                arch = arch.substr(pos + 1);
            }
            config.layer_sizes.push_back(std::stoi(arch));
        }
        else if (arg == "--data" && i + 1 < argc) {
            config.data_dir = argv[++i];
        }
        else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoi(argv[++i]);
        }
    }

    return config;
}
