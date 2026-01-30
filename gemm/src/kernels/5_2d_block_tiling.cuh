#pragma once

#include <cassert>
#include <cuda_runtime.h>

/**
A = MxK
B = KxN
C = MxN
*/

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void __launch_bounds__((BM*BN) / (TM*TN), 1) sgemm_2d_block_tiling(
    int M, int N, int K,
    float alpha, const float *A, const float *B,
    float beta, float *C) {
    
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlockTile = BM * BN;
    // a thread is responsible for calculating TM*TN results in a blocktile
    const uint numThreadsBlockTile = totalResultsBlockTile / (TM * TN);

    assert(numThreadsBlockTile == blockDim.x);

    // BN/TN are the no. of threads to span a column
    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow*BM*N + cCol*BN;

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    // calculates the no. of rows of As and Bs that are being loaded in a single step by a single block
    const uint strideA = numThreadsBlockTile / BK;
    const uint strideB = numThreadsBlockTile / BN;

    float threadResults[TM * TN] = {0.0};
    // register caches for A and B
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};

    for (int bkIdx{0}; bkIdx<K; bkIdx+=BK) {
        // loading data onto smem
        for (uint loadOffset{0}; loadOffset<BM; loadOffset+=strideA) {
            As[(innerRowA + loadOffset)*BK + innerColA] = A[(innerRowA + loadOffset)*K + innerColA];
        }
        for (uint loadOffset{0}; loadOffset<BK; loadOffset+=strideB) {
            Bs[(innerRowB + loadOffset)*BN + innerColB] = B[(innerRowB + loadOffset)*N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK*N;

        // calculate per thread results
        for (int dotIdx{0}; dotIdx<BK; dotIdx++) {
            // shared -> registers
            for (int i{0}; i<TM; i++) {
                regA[i] = As[(threadRow*TM + i)*BK + dotIdx];
            }
            for (int i{0}; i<TN; i++) {
                regB[i] = Bs[dotIdx*BN + threadCol*TN + i];
            }

            for (int resIdxM{0}; resIdxM<TM; resIdxM++) {
                for (int resIdxN{0}; resIdxN<TN; resIdxN++) {
                    threadResults[resIdxM*TN + resIdxN] += regA[resIdxM] * regB[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    // writing the result
    for (int resIdxM{0}; resIdxM<TM; resIdxM++) {
        for (int resIdxN{0}; resIdxN<TN; resIdxN++) {
            C[(threadRow*TM + resIdxM)*N + threadCol*TN + resIdxN] = 
                alpha * threadResults[resIdxM*TN + resIdxN] +
                beta  * C[(threadRow*TM + resIdxM)*N + threadCol*TN + resIdxN];
        }
    }

}
