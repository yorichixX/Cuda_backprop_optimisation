// ============================================================================
// V4: CUDA Streams + Pinned Memory — Overlap Transfer & Compute
// ============================================================================
//
// KEY IMPROVEMENT OVER V3:
//   V1-V3 copy ALL training data to GPU at startup. This works for MNIST
//   (~180MB) but fails for larger datasets (ImageNet = 150GB).
//
//   V4 demonstrates the production-grade approach:
//     1. Keep training data in HOST memory (not GPU memory)
//     2. Transfer mini-batches to GPU on-demand
//     3. OVERLAP the transfer of batch N+1 with the compute of batch N
//
// HOW OVERLAP WORKS:
//   CUDA streams are independent command queues. Operations on different
//   streams can execute concurrently (if the hardware supports it).
//
//   We use TWO streams:
//     - compute_stream: runs forward/backward/update kernels
//     - transfer_stream: runs cudaMemcpyAsync (H2D data transfers)
//
//   Timeline without streams:
//     [Transfer B0] [Compute B0] [Transfer B1] [Compute B1] ...
//
//   Timeline WITH streams:
//     [Transfer B0] [Compute B0      ] [Compute B1      ] ...
//                   [Transfer B1     ] [Transfer B2     ] ...
//                    ↑ overlapped!      ↑ overlapped!
//
// PINNED (PAGE-LOCKED) MEMORY:
//   Normal malloc'd memory can be swapped to disk by the OS. Before a DMA
//   transfer, the GPU driver must:
//     1. Pin the pages (prevent swapping)
//     2. Do the DMA transfer
//     3. Unpin the pages
//   This pinning overhead makes async transfers synchronous in practice.
//
//   cudaMallocHost() allocates page-locked memory that's ALWAYS resident
//   in physical RAM. DMA transfers are truly asynchronous — the CPU is
//   free to do other work while the GPU transfers data.
//
//   Downside: pinned memory is a limited OS resource. Don't pin everything.
//
// DOUBLE BUFFERING:
//   We allocate TWO GPU input buffers. While computing on buffer[0],
//   the next batch is being transferred into buffer[1]. Next iteration,
//   roles swap. This ensures there's always a buffer ready for the GPU.
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
// Kernels (same as V3 — fused forward + tiled backward)
// ============================================================================

__global__ void fused_gemm_bias_sigmoid(
        const float* W, const float* X, const float* bias,
        float* Z, float* A, int M, int N, int K) {
    __shared__ float tW[TILE][TILE+1], tX[TILE][TILE+1];
    int r = blockIdx.y*TILE+threadIdx.y, c = blockIdx.x*TILE+threadIdx.x;
    float sum = 0;
    for (int t = 0; t < div_ceil(K,TILE); t++) {
        int wc = t*TILE+threadIdx.x;
        tW[threadIdx.y][threadIdx.x] = (r<M && wc<K) ? W[r+wc*M] : 0;
        int xr = t*TILE+threadIdx.y;
        tX[threadIdx.y][threadIdx.x] = (xr<K && c<N) ? X[xr+c*K] : 0;
        __syncthreads();
        for (int i = 0; i < TILE; i++) sum += tW[threadIdx.y][i]*tX[i][threadIdx.x];
        __syncthreads();
    }
    if (r<M && c<N) {
        int idx = r+c*M;
        float z = sum + bias[r];
        Z[idx] = z;
        z = fmaxf(-30.0f, fminf(30.0f, z));
        A[idx] = 1.0f/(1.0f+expf(-z));
    }
}

__global__ void fused_gemm_bias_relu(
        const float* W, const float* X, const float* bias,
        float* Z, float* A, int M, int N, int K) {
    __shared__ float tW[TILE][TILE+1], tX[TILE][TILE+1];
    int r = blockIdx.y*TILE+threadIdx.y, c = blockIdx.x*TILE+threadIdx.x;
    float sum = 0;
    for (int t = 0; t < div_ceil(K,TILE); t++) {
        int wc = t*TILE+threadIdx.x;
        tW[threadIdx.y][threadIdx.x] = (r<M && wc<K) ? W[r+wc*M] : 0;
        int xr = t*TILE+threadIdx.y;
        tX[threadIdx.y][threadIdx.x] = (xr<K && c<N) ? X[xr+c*K] : 0;
        __syncthreads();
        for (int i = 0; i < TILE; i++) sum += tW[threadIdx.y][i]*tX[i][threadIdx.x];
        __syncthreads();
    }
    if (r<M && c<N) {
        int idx = r+c*M; float z = sum+bias[r];
        Z[idx]=z; A[idx]=fmaxf(0.0f,z);
    }
}

__global__ void tiled_gemm_tn(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tA[TILE][TILE+1], tB[TILE][TILE+1];
    int r=blockIdx.y*TILE+threadIdx.y, c=blockIdx.x*TILE+threadIdx.x;
    float sum=0;
    for (int t=0; t<div_ceil(K,TILE); t++) {
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

__global__ void tiled_gemm_nt_update(const float* A, const float* B, float* C,
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

__global__ void output_delta_sigmoid_kernel(float* d, const float* o, const float* t, int N, float iB) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float oi=o[i]; d[i]=(oi-t[i])*oi*(1-oi)*iB;}
}
__global__ void sigmoid_backward_kernel(float* d, const float* a, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float ai=a[i]; d[i]*=ai*(1-ai);}
}
__global__ void relu_backward_kernel(float* d, const float* z, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) d[i]*=(z[i]>0)?1.0f:0.0f;
}
__global__ void bias_update_kernel(float* b, const float* d, float lr, int M, int B) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<M){float g=0; for(int j=0;j<B;j++) g+=d[i+j*M]; b[i]-=lr*g;}
}

// ============================================================================
// MLP with Streams + Pinned Memory
// ============================================================================

struct MLP_GPU {
    MLPConfig config;
    int num_layers;

    // Network parameters (always on GPU)
    std::vector<float*> d_weights, d_biases, d_z, d_delta;
    // Activations — a[0] points to double-buffer input, rest are persistent
    std::vector<float*> d_a;

    // Double-buffered input/target on GPU
    float* d_input[2];
    float* d_target[2];

    // Pinned host memory (page-locked for async DMA)
    float* h_images_pinned;
    float* h_labels_pinned;
    int num_train_samples;

    // CUDA streams
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;

    float* d_loss;

    MLP_GPU(const MLPConfig& cfg, const MNISTData& td) : config(cfg) {
        num_layers = config.num_weight_layers();
        int B = config.batch_size;
        int id = config.layer_sizes[0], od = config.layer_sizes.back();
        num_train_samples = td.num_samples;

        // ---- Create streams ----
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&transfer_stream));

        // ---- Allocate pinned host memory ----
        // Page-locked memory enables truly asynchronous H2D transfers
        CUDA_CHECK(cudaMallocHost(&h_images_pinned, td.num_samples * id * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_labels_pinned, td.num_samples * od * sizeof(float)));
        memcpy(h_images_pinned, td.images.data(), td.num_samples * id * sizeof(float));
        memcpy(h_labels_pinned, td.labels.data(), td.num_samples * od * sizeof(float));

        // ---- Double buffers on GPU ----
        for (int i = 0; i < 2; i++) {
            CUDA_CHECK(cudaMalloc(&d_input[i],  id * B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_target[i], od * B * sizeof(float)));
        }

        // ---- Network parameters ----
        std::mt19937 rng(config.seed);
        d_weights.resize(num_layers); d_biases.resize(num_layers);
        d_z.resize(num_layers); d_delta.resize(num_layers);
        d_a.resize(num_layers + 1);

        for (int l = 0; l < num_layers; l++) {
            int ni = config.layer_sizes[l], no = config.layer_sizes[l+1];
            float lim = std::sqrt(6.0f / (ni + no));
            std::uniform_real_distribution<float> dist(-lim, lim);
            std::vector<float> hw(no*ni);
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

        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    }

    ~MLP_GPU() {
        for (int l = 0; l < num_layers; l++) {
            cudaFree(d_weights[l]); cudaFree(d_biases[l]);
            cudaFree(d_z[l]); cudaFree(d_delta[l]);
        }
        for (int l = 1; l <= num_layers; l++) cudaFree(d_a[l]);
        for (int i = 0; i < 2; i++) { cudaFree(d_input[i]); cudaFree(d_target[i]); }
        cudaFreeHost(h_images_pinned);
        cudaFreeHost(h_labels_pinned);
        cudaFree(d_loss);
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(transfer_stream);
    }

    // Forward pass with stream parameter (all kernels launch on this stream)
    void forward_on_stream(int B, cudaStream_t stream) {
        for (int l = 0; l < num_layers; l++) {
            int ni = config.layer_sizes[l], no = config.layer_sizes[l+1];
            bool io = (l == num_layers-1);
            dim3 bk(TILE, TILE); dim3 gg(div_ceil(B, TILE), div_ceil(no, TILE));
            if (io || config.hidden_activation == ActivationType::SIGMOID)
                fused_gemm_bias_sigmoid<<<gg, bk, 0, stream>>>(
                    d_weights[l], d_a[l], d_biases[l], d_z[l], d_a[l+1], no, B, ni);
            else
                fused_gemm_bias_relu<<<gg, bk, 0, stream>>>(
                    d_weights[l], d_a[l], d_biases[l], d_z[l], d_a[l+1], no, B, ni);
        }
    }

    void backward_update_on_stream(float* d_tgt, int B, cudaStream_t stream) {
        float lr = config.learning_rate, inv_B = 1.0f/B;
        int od = config.layer_sizes.back();

        output_delta_sigmoid_kernel<<<div_ceil(od*B,BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            d_delta[num_layers-1], d_a[num_layers], d_tgt, od*B, inv_B);

        for (int l = num_layers-2; l >= 0; l--) {
            int nt=config.layer_sizes[l+1], nn=config.layer_sizes[l+2];
            dim3 bk(TILE,TILE); dim3 gg(div_ceil(B,TILE),div_ceil(nt,TILE));
            tiled_gemm_tn<<<gg,bk,0,stream>>>(d_weights[l+1],d_delta[l+1],d_delta[l],nt,B,nn);
            int g1=div_ceil(nt*B,BLOCK_SIZE);
            if (config.hidden_activation == ActivationType::SIGMOID)
                sigmoid_backward_kernel<<<g1,BLOCK_SIZE,0,stream>>>(d_delta[l],d_a[l+1],nt*B);
            else
                relu_backward_kernel<<<g1,BLOCK_SIZE,0,stream>>>(d_delta[l],d_z[l],nt*B);
        }

        for (int l = 0; l < num_layers; l++) {
            int ni=config.layer_sizes[l], no=config.layer_sizes[l+1];
            dim3 bk(TILE,TILE); dim3 gg(div_ceil(ni,TILE),div_ceil(no,TILE));
            tiled_gemm_nt_update<<<gg,bk,0,stream>>>(d_delta[l],d_a[l],d_weights[l],no,ni,B,-lr,1.0f);
            bias_update_kernel<<<div_ceil(no,BLOCK_SIZE),BLOCK_SIZE,0,stream>>>(d_biases[l],d_delta[l],lr,no,B);
        }
    }

    // ---- Train one epoch with overlapped transfers ----
    float train_epoch() {
        int B = config.batch_size;
        int id = config.layer_sizes[0], od = config.layer_sizes.back();
        int num_batches = num_train_samples / B;
        float total_loss = 0;

        // Pre-load first batch synchronously
        CUDA_CHECK(cudaMemcpy(d_input[0], h_images_pinned,
                              id*B*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_target[0], h_labels_pinned,
                              od*B*sizeof(float), cudaMemcpyHostToDevice));

        for (int b = 0; b < num_batches; b++) {
            int cur = b % 2;
            int nxt = 1 - cur;

            // ---- Async transfer of NEXT batch (overlaps with compute) ----
            if (b + 1 < num_batches) {
                int next_off = (b + 1) * B;
                CUDA_CHECK(cudaMemcpyAsync(d_input[nxt],
                    h_images_pinned + next_off * id,
                    id*B*sizeof(float), cudaMemcpyHostToDevice, transfer_stream));
                CUDA_CHECK(cudaMemcpyAsync(d_target[nxt],
                    h_labels_pinned + next_off * od,
                    od*B*sizeof(float), cudaMemcpyHostToDevice, transfer_stream));
            }

            // ---- Compute on current batch ----
            d_a[0] = d_input[cur];
            forward_on_stream(B, compute_stream);

            // Loss computation
            CUDA_CHECK(cudaMemsetAsync(d_loss, 0, sizeof(float), compute_stream));
            mse_loss_kernel<<<div_ceil(od*B, BLOCK_SIZE), BLOCK_SIZE,
                              BLOCK_SIZE*sizeof(float), compute_stream>>>(
                d_a[num_layers], d_target[cur], d_loss, od*B);

            backward_update_on_stream(d_target[cur], B, compute_stream);

            // ---- Sync both streams before next iteration ----
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            CUDA_CHECK(cudaStreamSynchronize(transfer_stream));

            // Read loss (already on CPU after sync)
            float loss_val;
            CUDA_CHECK(cudaMemcpy(&loss_val, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            total_loss += loss_val / (2.0f * od * B);
        }
        return total_loss / num_batches;
    }

    float evaluate(const MNISTData& td) {
        int id=config.layer_sizes[0], od=config.layer_sizes.back();
        int B=config.batch_size, correct=0;
        float* dt; CUDA_CHECK(cudaMalloc(&dt, id*B*sizeof(float)));
        std::vector<float> ho(od*B);
        for (int s=0;s<td.num_samples;s+=B) {
            int aB=std::min(B,td.num_samples-s);
            CUDA_CHECK(cudaMemcpy(dt, td.get_image(s), id*aB*sizeof(float), cudaMemcpyHostToDevice));
            float* sv=d_a[0]; d_a[0]=dt;
            forward_on_stream(aB, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(ho.data(), d_a[num_layers], od*aB*sizeof(float), cudaMemcpyDeviceToHost));
            for (int i=0;i<aB;i++) {
                int p=0,a=0; float mp=-1e30f,ml=-1e30f;
                for (int c=0;c<od;c++) {
                    if(ho[c+i*od]>mp){mp=ho[c+i*od];p=c;}
                    if(td.labels[(s+i)*od+c]>ml){ml=td.labels[(s+i)*od+c];a=c;}
                }
                if(p==a) correct++;
            }
            d_a[0]=sv;
        }
        cudaFree(dt);
        return (float)correct/td.num_samples;
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    printf("================================================================\n");
    printf("  V4: CUDA Streams + Pinned Memory (Overlapped Pipeline)\n");
    printf("================================================================\n");

    MLPConfig config = parse_args(argc, argv);
    config.print();

    MNISTData train = load_mnist(config.data_dir+"/train-images-idx3-ubyte",
                                  config.data_dir+"/train-labels-idx1-ubyte");
    MNISTData test = load_mnist(config.data_dir+"/t10k-images-idx3-ubyte",
                                 config.data_dir+"/t10k-labels-idx1-ubyte");

    if (config.layer_sizes.front()!=train.image_size||config.layer_sizes.back()!=train.num_classes) {
        fprintf(stderr,"Error: Architecture mismatch\n"); return 1;
    }

    MLP_GPU mlp(config, train);
    double tt = 0;
    int nb = train.num_samples / config.batch_size;

    printf("\nTraining: %d samples, %d batches/epoch\n\n", train.num_samples, nb);

    // Warm-up
    mlp.d_a[0] = mlp.d_input[0];
    mlp.forward_on_stream(config.batch_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CPUTimer timer; CUDATimer ctimer;
    for (int ep = 0; ep < config.epochs; ep++) {
        timer.start(); ctimer.start();
        float loss = mlp.train_epoch();
        float gt = ctimer.stop(); double wt = timer.elapsed_ms(); tt += wt;
        float acc = mlp.evaluate(test);
        printf("Epoch %2d/%d | Loss: %.6f | Accuracy: %6.2f%% | GPU: %8.1f ms | Wall: %8.1f ms\n",
               ep+1, config.epochs, loss, acc*100, gt, wt);
    }

    printf("\n================================================================\n");
    printf("  Results — V4 Streams + Pinned Memory\n");
    printf("================================================================\n");
    printf("Total training time:   %10.1f ms\n", tt);
    printf("Average epoch time:    %10.1f ms\n", tt / config.epochs);
    printf("================================================================\n");
    return 0;
}
