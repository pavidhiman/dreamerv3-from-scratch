#include <cuda_runtime.h>
#include <stdio.h>

// A is the left matrix (shape M x K), B is the right matrix (shape K x N), C is the output (shape M x N)
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // col comps
    int col = blockIdx.x * blockDim.x + threadIdx.x; // row comps

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) { // dot product 
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" {
void cuda_matmul(float *host_A, float *host_B, float *host_C, int M, int K, int N) {
    float *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B, size_B, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(host_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}
