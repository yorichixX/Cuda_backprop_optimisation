#pragma once

// ============================================================================
// MNIST Data Loader
// ============================================================================
// Reads the raw MNIST IDX binary format (no external library dependencies).
//
// Data layout in memory (critical for GPU performance):
//   images: [img0_px0, img0_px1, ..., img0_px783, img1_px0, ...]
//   labels: [lbl0_c0, lbl0_c1, ..., lbl0_c9, lbl1_c0, ...]
//
// This layout means each sample's data is contiguous. When we take a batch
// of B consecutive samples starting at index i:
//   batch_images = &images[i * 784]  → shape (784, B) in column-major
//   batch_labels = &labels[i * 10]   → shape (10, B) in column-major
//
// Why column-major? cuBLAS (and BLAS in general) expects column-major matrices.
// By storing each sample as a "column" of pixels, we can pass batch pointers
// directly to cuBLAS without any transposition. This zero-copy design is
// critical for avoiding unnecessary data movement.
// ============================================================================

#include <vector>
#include <string>

struct MNISTData {
    std::vector<float> images;  // num_samples * 784, normalized to [0, 1]
    std::vector<float> labels;  // num_samples * 10,  one-hot encoded
    int num_samples = 0;
    int image_size  = 784;      // 28 * 28
    int num_classes = 10;       // digits 0-9

    // Get pointer to start of image i (784 contiguous floats)
    const float* get_image(int i) const {
        return images.data() + i * image_size;
    }

    // Get pointer to start of label i (10 contiguous floats, one-hot)
    const float* get_label(int i) const {
        return labels.data() + i * num_classes;
    }

    // Get pointer to a batch of B images starting at index i
    // Returns pointer to (784, B) column-major matrix
    const float* get_image_batch(int i) const {
        return images.data() + i * image_size;
    }

    // Get pointer to a batch of B labels starting at index i
    // Returns pointer to (10, B) column-major matrix
    const float* get_label_batch(int i) const {
        return labels.data() + i * num_classes;
    }
};

// Load MNIST data from IDX binary files
// image_path: path to *-images-idx3-ubyte file
// label_path: path to *-labels-idx1-ubyte file
MNISTData load_mnist(const std::string& image_path, const std::string& label_path);
