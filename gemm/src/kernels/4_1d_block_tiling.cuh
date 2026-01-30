#pragma once

#include <cassert>
#include <cuda_runtime.h>

/**
A = MxK
B = KxN
C = MxN
*/

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_block_tiling(int M, int N, int K,
                            float alpha, const float *A, const float *B,
                            float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // to get to start index of the new row/col which needs to be tiled.
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow*BM*N + cCol*BN;

    assert(BM*BK == blockDim.x);
    assert(BN*BK == blockDim.x);

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.0};

    for (int bkIdx{0}; bkIdx<K; bkIdx+=BK){
        As[innerRowA*BK + innerColA] = A[innerRowA*K + innerColA];
        Bs[innerRowB*BN + innerColB] = B[innerRowB*N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx{0}; dotIdx<BK; dotIdx++) {
            float tmpB = Bs[dotIdx*BN + threadCol];
            for (int resIdx{0}; resIdx<TM; resIdx++) {
                threadResults[resIdx] += As[(threadRow*TM + resIdx)*BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (int resIdx{0}; resIdx<TM; resIdx++) {
        C[(threadRow*TM + resIdx)*N + threadCol] = 
            alpha*threadResults[resIdx] + 
            beta*C[(threadRow*TM + resIdx)*N + threadCol];
    }

}
