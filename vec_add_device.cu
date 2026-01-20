#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = x; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void init_vector(float *vec, int n) {
  for(int i=0; i<n; i++) {
    vec[i] = (float)rand() / RAND_MAX;
  }
}

__global__ void sum_vector(float *d_a, float *d_b, float *d_c, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n)
        d_c[i] = d_a[i] + d_b[i];
}

__global__ void sum_vector_easy(float *d_a, float *d_b, float *d_c){
    int i = threadIdx.x;
    d_c[i] = d_a[i] + d_b[i];
}

int main() {

    float *h_A, *h_B;
    float *d_A, *d_B ,*d_C;
    int size = N * sizeof(float);

    // alloting memory in cpu
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);

    // alloting memory in GPU
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMalloc(&d_A, size));
    // generating random values in vector
    init_vector(h_A, N);
    init_vector(h_B, N);
    
    // tranfering data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // calling kernel
    sum_vector <<< ceil(N/256.0), 256 >>> (d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // freeing device memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
