// ============================================================================
// V0: CPU Baseline — Main Entry Point
// ============================================================================
// Usage:
//   v0_cpu [--arch 784,512,256,10] [--lr 0.01] [--epochs 20] [--batch 256]
//          [--activation relu|sigmoid] [--data ./data]
//
// Trains a fully-connected MLP on MNIST using single-threaded CPU.
// This is the sequential baseline against which all GPU versions are compared.
// ============================================================================

#include "mlp_cpu.h"
#include "data_loader.h"
#include "timer.h"
#include <cstdio>
#include <iostream>

int main(int argc, char* argv[]) {
    printf("================================================================\n");
    printf("  V0: CPU Baseline (Sequential Single-Threaded)\n");
    printf("================================================================\n");

    // Parse command-line arguments
    MLPConfig config = parse_args(argc, argv);
    config.print();

    // Load MNIST dataset
    printf("Loading MNIST data from '%s'...\n", config.data_dir.c_str());
    MNISTData train_data = load_mnist(
        config.data_dir + "/train-images-idx3-ubyte",
        config.data_dir + "/train-labels-idx1-ubyte");
    MNISTData test_data = load_mnist(
        config.data_dir + "/t10k-images-idx3-ubyte",
        config.data_dir + "/t10k-labels-idx1-ubyte");
    printf("\n");

    // Validate architecture matches data
    if (config.layer_sizes.front() != train_data.image_size) {
        fprintf(stderr, "Error: input layer size (%d) != image size (%d)\n",
                config.layer_sizes.front(), train_data.image_size);
        return 1;
    }
    if (config.layer_sizes.back() != train_data.num_classes) {
        fprintf(stderr, "Error: output layer size (%d) != num classes (%d)\n",
                config.layer_sizes.back(), train_data.num_classes);
        return 1;
    }

    // Create the MLP
    MLP_CPU mlp(config);

    int num_batches = train_data.num_samples / config.batch_size;
    double total_train_time = 0.0;

    printf("Training: %d samples, %d batches/epoch, batch_size=%d\n\n",
           train_data.num_samples, num_batches, config.batch_size);

    // ---- Training Loop ----
    CPUTimer epoch_timer;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        epoch_timer.start();
        float total_loss = 0.0f;

        for (int b = 0; b < num_batches; b++) {
            int offset = b * config.batch_size;
            const float* imgs = train_data.get_image(offset);
            const float* lbls = train_data.get_label(offset);

            total_loss += mlp.train_batch(imgs, lbls, config.batch_size);
        }

        double epoch_time = epoch_timer.elapsed_ms();
        total_train_time += epoch_time;

        // Evaluate on test set
        float accuracy = mlp.evaluate(
            test_data.images.data(), test_data.labels.data(),
            test_data.num_samples);

        printf("Epoch %2d/%d | Loss: %.6f | Accuracy: %6.2f%% | Time: %8.1f ms\n",
               epoch + 1, config.epochs,
               total_loss / num_batches,
               accuracy * 100.0f,
               epoch_time);
    }

    // ---- Summary ----
    printf("\n================================================================\n");
    printf("  Results — V0 CPU Baseline\n");
    printf("================================================================\n");
    printf("Total training time:   %10.1f ms\n", total_train_time);
    printf("Average epoch time:    %10.1f ms\n", total_train_time / config.epochs);
    printf("================================================================\n");

    return 0;
}
