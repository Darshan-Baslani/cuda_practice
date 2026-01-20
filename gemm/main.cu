#include <iostream>
#include <cuda_runtime.h>

static constexpr int CEIL_DIV (int a, int b) {
  return (a + b - 1) / b;
}


int main() {
  constexpr int M=128, N=7168, K=16384;
  constexpr int alpha=1, beta=0;

  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  dim3 blockDim(32, 32, 1);

  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

  return 0;
}
