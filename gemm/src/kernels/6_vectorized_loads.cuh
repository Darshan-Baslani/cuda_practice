#pragma once

#include <cassert>
#include <cuda_runtime.h>

/**
A = MxK
B = KxN
C = MxN
*/

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void __launch_bounds__((BM*BN) / (TM*TN), 1) sgemm_vectorized_loads(
    int M, int N, int K,
    float alpha, float *A, float *B, // we are not doing const loads anymore of A and B b.c. we would be casting them to float 4 :)
    float beta, float *C) {
    
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // BN/TN are the no. of threads to span a column
    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    const uint extraCols = 5; // i am not sure why we are doing this!
    __shared__ float Bs[BK * (BN + extraCols)];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow*BM*N + cCol*BN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit/32bit = 4 elements per thread at each step
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);

    float threadResults[TM * TN] = {0.0};
    // register caches for A and B
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};

    for (int bkIdx{0}; bkIdx<K; bkIdx+=BK) {
        // loading data onto smem (vectorized loads!)
        float4 tmp = 
            reinterpret_cast<float4 *>(&A[innerRowA*K + innerColA * 4])[0];
        As[(innerColA*4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA*4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA*4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA*4 + 3) * BM + innerRowA] = tmp.w;

        tmp = reinterpret_cast<float4 *>(&B[innerRowB*N + innerColB * 4])[0];
        Bs[innerRowB*(BN + extraCols) + innerColB*4 + 0] = tmp.x;
        Bs[innerRowB*(BN + extraCols) + innerColB*4 + 1] = tmp.y;
        Bs[innerRowB*(BN + extraCols) + innerColB*4 + 2] = tmp.z;
        Bs[innerRowB*(BN + extraCols) + innerColB*4 + 3] = tmp.w;
        __syncthreads();

        A += BK;
        B += BK*N;

        // calculate per thread results
        for (int dotIdx{0}; dotIdx<BK; dotIdx++) {
            // shared -> registers
            for (int i{0}; i<TM; i++) {
                regA[i] = As[dotIdx*BM + threadRow*TM + i];
            }
            for (int i{0}; i<TN; i++) {
                regB[i] = Bs[dotIdx*(BN + extraCols) + threadCol*TN + i];
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
            for (int resIdxN{0}; resIdxN<TN; resIdxN+=4) {
                float4 tmp = reinterpret_cast<float4 *>(
                    &C[(threadRow*TM + resIdxM)*N + threadCol*TN + resIdxN]
                )[0];

                tmp.x = alpha*threadResults[resIdxM*TN + resIdxN + 0] + beta*tmp.x;
                tmp.y = alpha*threadResults[resIdxM*TN + resIdxN + 1] + beta*tmp.y;
                tmp.z = alpha*threadResults[resIdxM*TN + resIdxN + 2] + beta*tmp.z;
                tmp.w = alpha*threadResults[resIdxM*TN + resIdxN + 3] + beta*tmp.w;

                reinterpret_cast<float4 *>(
                    &C[(threadRow*TM + resIdxM)*N + threadCol*TN + resIdxN]
                )[0] = tmp;
            }
        }
}
