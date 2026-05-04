#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gru_forward_kernel(
    float *x, float *h_prev,
    float *W_r, float *b_r,
    float *W_z, float *b_z,
    float *W_c, float *b_c,
    float *h_out,
    int batch, int input_size, int hidden_size
) {
    int b = blockIdx.x;
    int j = threadIdx.x;

    if (b >= batch || j >= hidden_size) return;

    int combined_size = input_size + hidden_size;

    float r_val = b_r[j]; // reset gate 
    for (int k = 0; k < combined_size; k++) {
        float inp = (k < input_size) ? x[b * input_size + k] : h_prev[b * hidden_size + (k - input_size)];
        r_val += inp * W_r[k * hidden_size + j];
    }
    r_val = sigmoid(r_val);

    float z_val = b_z[j]; // update gate 
    for (int k = 0; k < combined_size; k++) {
        float inp = (k < input_size) ? x[b * input_size + k] : h_prev[b * hidden_size + (k - input_size)];
        z_val += inp * W_z[k * hidden_size + j];
    }
    z_val = sigmoid(z_val);

    float c_val = b_c[j]; // candidate gate 
    for (int k = 0; k < combined_size; k++) {
        float inp;
        if (k < input_size)
            inp = x[b * input_size + k];
        else
            inp = r_val * h_prev[b * hidden_size + (k - input_size)];
        c_val += inp * W_c[k * hidden_size + j];
    }
    c_val = tanhf(c_val);

    h_out[b * hidden_size + j] = z_val * h_prev[b * hidden_size + j] + (1.0f - z_val) * c_val; // final hidden state
    // z=1 => keep old h, z=0 => use candidate
}

extern "C" {

void cuda_gru_forward(
    float *x, float *h_prev,
    float *W_r, float *b_r,
    float *W_z, float *b_z,
    float *W_c, float *b_c,
    float *h_out,
    int batch, int input_size, int hidden_size
) {
    float *d_x, *d_h_prev, *d_h_out;
    float *d_W_r, *d_b_r, *d_W_z, *d_b_z, *d_W_c, *d_b_c;

    int combined_size = input_size + hidden_size;

    size_t sz_x = batch * input_size * sizeof(float);
    size_t sz_h = batch * hidden_size * sizeof(float);
    size_t sz_W = combined_size * hidden_size * sizeof(float);
    size_t sz_b = hidden_size * sizeof(float);

    cudaMalloc(&d_x, sz_x);
    cudaMalloc(&d_h_prev, sz_h);
    cudaMalloc(&d_h_out, sz_h);
    cudaMalloc(&d_W_r, sz_W);
    cudaMalloc(&d_b_r, sz_b);
    cudaMalloc(&d_W_z, sz_W);
    cudaMalloc(&d_b_z, sz_b);
    cudaMalloc(&d_W_c, sz_W);
    cudaMalloc(&d_b_c, sz_b);

    cudaMemcpy(d_x, x, sz_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h_prev, h_prev, sz_h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_r, W_r, sz_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_r, b_r, sz_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_z, W_z, sz_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_z, b_z, sz_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_c, W_c, sz_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_c, b_c, sz_b, cudaMemcpyHostToDevice);

    dim3 blocks(batch);
    dim3 threads(hidden_size);

    gru_forward_kernel<<<blocks, threads>>>(
        d_x, d_h_prev,
        d_W_r, d_b_r,
        d_W_z, d_b_z,
        d_W_c, d_b_c,
        d_h_out,
        batch, input_size, hidden_size
    );

    cudaMemcpy(h_out, d_h_out, sz_h, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_h_prev);
    cudaFree(d_h_out);
    cudaFree(d_W_r);
    cudaFree(d_b_r);
    cudaFree(d_W_z);
    cudaFree(d_b_z);
    cudaFree(d_W_c);
    cudaFree(d_b_c);
}

}
