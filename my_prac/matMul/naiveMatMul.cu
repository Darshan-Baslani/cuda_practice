#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void naiveMatMul(float *M, float *N, float *O, int width) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < width && col < width) {
    float val = 0;
    for (int k = 0; k < width; ++k) {
      val += M[row * width + k] * N[k * width + col];
    }
    O[row * width + col] = val;
  }
}

__global__ void rowMatMul(float *M, float *N, float *O, int width) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < width && col < width) {
    float *val;
    float **valAddr = &val;
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < width; ++j) {
        val += M[row * width + j] * N[j * width + col];
      }
      ++val;
    }
  }
}

__global__ void expMatMul(float *M, float *N, float *O, int width) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  float vals[width][width];
  if (row < width && col < width) {
    for (int i = 0; i < width; i++) {
      if (row == i) {
      }
    }
  }
}
