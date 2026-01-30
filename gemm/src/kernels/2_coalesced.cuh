#pragma once

#include <cuda_runtime.h>

/**
A = MxK
B = KxN
C = MxN
*/

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K,
                            float alpha, const float *A, const float *B,
                            float beta, float *C) {
    
    const uint cCol = blockIdx.x * BLOCKSIZE + threadIdx.x;
    const uint cRow = blockIdx.y * BLOCKSIZE + threadIdx.y;
    if (cRow < M && cCol < N) {
        float acc = 0.0;
        for (int k{0}; k<K; k++) {
            acc += A[cRow*K + k] * B[k*N + cCol];
        }
        C[cRow*N + cCol] = alpha*acc + beta*C[cRow*N + cCol];
    }

}