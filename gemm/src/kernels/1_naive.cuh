#pragma once

#include <cuda_runtime.h>

/**
A = MxK
B = KxN
C = MxN
*/

__global__ void sgemm_naive(int M, int N, int K,
                            float alpha, const float *A, const float *B,
                            float beta, float *C) {
    const uint i = blockIdx.y * blockDim.y + threadIdx.y;
    const uint j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < M) {
        float acc = 0.0;
        for (int k{0}; k < K; k++) {
            acc += A[j * K + k] * B[N * k + i];
        }
        C[j * N + i] = alpha * acc + beta * C[j * N + i];
    }
}
