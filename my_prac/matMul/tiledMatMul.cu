#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>

// #define WIDTH 6144
#define Mrow 1024
#define McolNrow 512
#define Ncol 256
#define MATRIX_SIZE (WIDTH*WIDTH)
#define TILE_WIDTH 32

#define CUDA_CHECK(res) {                                                      \;
    gpuAssert((res), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void tiledMatMul(float *M, float *N, float *O) {
  __shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Ns[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = TILE_WIDTH*by + ty;
  int col = TILE_WIDTH*bx + tx;

  float o_value = 0.0f;
  for (int tile_idx = 0; tile_idx < ((McolNrow + TILE_WIDTH - 1)/TILE_WIDTH); tile_idx++) {
    if (row < Mrow && (tile_idx*TILE_WIDTH + tx) < McolNrow) 
      Ms[ty][tx] = M[row*McolNrow + tile_idx*TILE_WIDTH + tx];
    else
      Ms[ty][tx] = 0.0f;

    if (col < Ncol && (tile_idx*TILE_WIDTH + ty) < McolNrow)
      Ns[ty][tx] = N[(tile_idx * TILE_WIDTH + ty) * Ncol + col];
    else
      Ns[ty][tx] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++) {
      o_value += Ms[ty][k] * Ns[k][tx];
    }
    __syncthreads();
  }
  if (row < Mrow && col < Ncol)
    O[row*Ncol + col] = o_value;
}

__global__ void naiveMatMul(float *M, float *N, float *O) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < Mrow && col < Ncol) {
    float val = 0;
    for (int k = 0; k < McolNrow; ++k) {
      val += M[row * McolNrow + k] * N[k * Ncol + col];
    }
    O[row * Ncol + col] = val;
  }
}

// // for the old kernel; can't use this in gemm
// void initializeMatrices(float *h_M, float *h_N) {
//   for (int i = 0; i < WIDTH * WIDTH; ++i) {
//     h_M[i] = static_cast<float>(i % 10);
//     h_N[i] = static_cast<float>(i % 7);
//   }
// }

void initializeM(float *h_M) {
  for (int i = 0; i < (Mrow * McolNrow); i++)
    h_M[i] = static_cast<float>(i % 10);
}

void initializeN(float *h_N) {
  for (int i = 0; i < (Ncol * McolNrow); i++)
    h_N[i] = static_cast<float>(i % 7);
}

bool compareMatrices(const float *A, const float *B, float epsilon = 1e-5f) {
  for (int i = 0; i < Mrow * Ncol; ++i) {
    if (fabs(A[i] - B[i]) > epsilon) {
      std::cerr << "Matrices differ at index " << i << "! A[" << i
                << "] = " << A[i] << ", B[" << i << "] = " << B[i] << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  size_t sizeM = Mrow * McolNrow;
  size_t sizeN = Ncol * McolNrow;
  size_t sizeO = Mrow * Ncol;

  float *h_M = new float[sizeM];
  float *h_N = new float[sizeN];
  float *h_O = new float[sizeO];
  float *h_O_correct = new float[sizeO];

  // initializeMatrices(h_M, h_N);
  initializeM(h_M);
  initializeN(h_N);

  float *d_M, *d_N, *d_O, *d_O_correct;
  CUDA_CHECK(cudaMalloc(&d_M, sizeM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_N, sizeN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_O, sizeO * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_O_correct, sizeO * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice));

  dim3 blockSize(16, 16);
  dim3 gridSize(
    (Ncol + blockSize.x - 1) / blockSize.x, 
    (Mrow + blockSize.y - 1) / blockSize.y
  );

  printf("M shape: (%d, %d), N shape: (%d, %d), O shape: (%d, %d), tile width: (%d, %d)\n", Mrow, McolNrow, McolNrow, Ncol, Mrow, Ncol, TILE_WIDTH, TILE_WIDTH);

  std::cout << "Launching naive matmul kernel" << std::endl;

  auto start_naive = std::chrono::high_resolution_clock::now();

  naiveMatMul <<<gridSize, blockSize>>>(d_M, d_N, d_O_correct);

  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_naive = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_naive = end_naive - start_naive;

  std::cout << "Successfully completed naive matmul" << std::endl;
  std::cout << "Time: " << duration_naive.count() << " ms" << std::endl;

  std::cout <<"starting tiled matmul kernel" << std::endl;

  dim3 blockDim (TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim ((int) Ncol / TILE_WIDTH, (int) Mrow / TILE_WIDTH);

  auto start_tiled = std::chrono::high_resolution_clock::now();

  tiledMatMul <<<gridDim, blockDim >>>(d_M, d_N, d_O);

  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_tiled = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> duration_tiled = end_tiled - start_tiled;

  std::cout << "Successfully completed tiled matmul" << std::endl;
  std::cout << "Time: " << duration_tiled.count() << " ms" << std::endl;



  CUDA_CHECK(cudaMemcpy(h_O, d_O, sizeO, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_O_correct, d_O_correct, sizeO, cudaMemcpyDeviceToHost));

  if (compareMatrices(h_O_correct, h_O)) {
    std::cout << "both matrices are same" << std::endl;
  } else {
    std::cerr << "both matrices aren't same" << std::endl;
  }

  cudaFree(d_M); cudaFree(d_N); cudaFree(d_O); cudaFree(d_O_correct);
}
