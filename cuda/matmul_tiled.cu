#include <cuda_runtime.h>
#include <stdio.h>

#define TILE  // tile size = 16x16 since each block of threads will load and process 16x16 chunks at a time

__global__ void matmul_tiled_kernel(float *A, float *B, float *C, int M, int K, int N) {
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;
    // cooperative loading where each thread loads 1 element of tile_A and 1 from tile_B
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads(); // makes all 256 threads in block wait 

        for (int k = 0; k < TILE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" {
void cuda_matmul_tiled(float *host_A, float *host_B, float *host_C, int M, int K, int N) {
    float *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B, size_B, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(host_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
}
