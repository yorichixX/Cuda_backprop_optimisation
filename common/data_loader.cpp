// ============================================================================
// MNIST IDX File Parser
// ============================================================================
// The IDX format stores data in big-endian binary. Structure:
//
// Image file:
//   [4 bytes magic=2051] [4 bytes num_images] [4 bytes rows] [4 bytes cols]
//   [rows*cols bytes per image × num_images]
//
// Label file:
//   [4 bytes magic=2049] [4 bytes num_labels]
//   [1 byte per label × num_labels]
//
// All multi-byte integers are stored in MSB-first (big-endian) order.
// x86/x64 is little-endian, so we need to byte-swap when reading.
// ============================================================================

#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Read a 32-bit unsigned integer in big-endian byte order
// ---------------------------------------------------------------------------
// Why manual byte swapping? We can't use ntohl() portably, and this is
// explicit about what's happening: bytes[0] is the most significant byte.
// ---------------------------------------------------------------------------
static uint32_t read_uint32_be(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8)  |
           (static_cast<uint32_t>(bytes[3]));
}

MNISTData load_mnist(const std::string& image_path, const std::string& label_path) {
    MNISTData data;

    // --------------- Load Images ---------------
    {
        std::ifstream f(image_path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open image file: " + image_path);
        }

        uint32_t magic = read_uint32_be(f);
        if (magic != 2051) {
            throw std::runtime_error(
                "Invalid image magic number: " + std::to_string(magic) +
                " (expected 2051)");
        }

        data.num_samples = static_cast<int>(read_uint32_be(f));
        int rows = static_cast<int>(read_uint32_be(f));
        int cols = static_cast<int>(read_uint32_be(f));
        data.image_size = rows * cols;  // Should be 784

        // Read raw pixel data (uint8) and normalize to [0, 1] (float32)
        // Normalization: pixel_float = pixel_uint8 / 255.0
        // This maps [0, 255] → [0.0, 1.0], which:
        //   1. Prevents sigmoid saturation (raw 0-255 would push sigmoid to 0 or 1)
        //   2. Keeps gradients in a reasonable range
        //   3. Makes learning rate less dependent on input scale
        size_t total_pixels = static_cast<size_t>(data.num_samples) * data.image_size;
        data.images.resize(total_pixels);

        std::vector<uint8_t> raw(total_pixels);
        f.read(reinterpret_cast<char*>(raw.data()), total_pixels);

        for (size_t i = 0; i < total_pixels; i++) {
            data.images[i] = static_cast<float>(raw[i]) / 255.0f;
        }

        std::cout << "Loaded " << data.num_samples << " images ("
                  << rows << "x" << cols << ")" << std::endl;
    }

    // --------------- Load Labels (one-hot encoded) ---------------
    {
        std::ifstream f(label_path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open label file: " + label_path);
        }

        uint32_t magic = read_uint32_be(f);
        if (magic != 2049) {
            throw std::runtime_error(
                "Invalid label magic number: " + std::to_string(magic) +
                " (expected 2049)");
        }

        int num_labels = static_cast<int>(read_uint32_be(f));
        if (num_labels != data.num_samples) {
            throw std::runtime_error(
                "Image/label count mismatch: " +
                std::to_string(data.num_samples) + " images vs " +
                std::to_string(num_labels) + " labels");
        }

        // One-hot encoding: label 3 → [0,0,0,1,0,0,0,0,0,0]
        // Why one-hot? The output layer has 10 sigmoid neurons, each
        // representing the probability of one digit. One-hot targets make
        // the MSE loss push each output toward either 0 or 1.
        data.num_classes = 10;
        data.labels.resize(static_cast<size_t>(data.num_samples) * data.num_classes, 0.0f);

        std::vector<uint8_t> raw_labels(data.num_samples);
        f.read(reinterpret_cast<char*>(raw_labels.data()), data.num_samples);

        for (int i = 0; i < data.num_samples; i++) {
            int label = static_cast<int>(raw_labels[i]);
            data.labels[static_cast<size_t>(i) * data.num_classes + label] = 1.0f;
        }

        std::cout << "Loaded " << num_labels << " labels ("
                  << data.num_classes << " classes)" << std::endl;
    }

    return data;
}
