#pragma once

#include <cuda_runtime.h>

/**
A = MxK
B = KxN
C = MxN
*/

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const uint TILESIZE>
__global__ void sgemm_share_mem(int M, int N, int K,
                            float alpha, const float *A, const float *B,
                            float beta, float *C) {

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[TILESIZE * TILESIZE];
    __shared__ float Bs[TILESIZE * TILESIZE];

    const uint threadCol = threadIdx.x % TILESIZE;
    const uint threadRow = threadIdx.x / TILESIZE;

    // advancing the starting pointer
    A += cRow * TILESIZE * K;                   // row=cRow, col=0
    B += cCol * TILESIZE;                       // row=0, col=cCol
    C += cRow * TILESIZE * K + cCol * TILESIZE; // row=cRow, col=cCol

    float acc = 0.0;
    for (int tileIdx{0}; tileIdx<K; tileIdx++) {
        As[threadRow*TILESIZE + threadCol] = A[threadRow*K + threadCol];
        Bs[threadRow*TILESIZE + threadCol] = B[threadRow*N + threadCol];

        __syncthreads();
        A += TILESIZE;
        B += TILESIZE*N;

        for (int k{0}; k<TILESIZE; k++) {
            acc += As[threadRow*TILESIZE + k] * Bs[k*TILESIZE + threadCol];
        }

        __syncthreads();
        C[threadRow*N + threadCol] = alpha*acc + beta*C[threadRow*N + threadCol];
    }
}