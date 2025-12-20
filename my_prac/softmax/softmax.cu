#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

__global__ void safe_softmax(float *x, float *y, int M, int N) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < M) {
    float x_max = -INFINITY;
    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      x_max = std::max(x_max, x[i]);
    }

    float d = 0.0f;
    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      d += expf(x[i] - x_max);
    }

    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      y[i] = expf(x[i] - x_max) / d;
    }
  }
}

__global__ void online_softmax(float *x, float *y, int M, int N) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < M) {
    float x_max = -INFINITY;
    float d = 1.0f;
    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      if (x[i] > x_max) {
        d = d * expf(x_max - x[i]);
        x_max = x[i];
      }
      d += expf(x[i] - x_max);
    }

    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      y[i] = expf(x[i] - x_max) / d;
    }
  }
}
