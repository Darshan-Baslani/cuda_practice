#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

// #define CUDA_CHECK(err)                                                        \
//   {                                                                            \
//     cudaError_t err_ = (err);                                                  \
//     if (err_ != cudaSuccess) {                                                 \
//       std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__     \
//                 << ": " << cudaGetErrorString(err_) << std::endl;              \
//       exit(EXIT_FAILURE);                                                      \
//     }                                                                          \
//   }

#define CUDA_CHECK(ans)                                                        \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
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

// my failed attempt for solving ch-3 q-1.1
__global__ void rowMatMul(float *M, float *N, float *O, int width) {
  int col = (int)(blockDim.x * blockIdx.x + threadIdx.x) % width;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < width && col < width) {
    while (true) {
      float val = 0;
      for (int k = 0; k < width; ++k) {
        val += M[row * width + k] * N[k * width + col];
      }
      O[row * width + col] = val;
      col++;
      if (col == width)
        break;
    }
  }
}

// correct approach for ch-3 1.1
__global__ void correctRowMatMul(float *M, float *N, float *O, int width) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < width) {
    for (int col = 0; col < width; ++col) {
      float val = 0;
      for (int k = 0; k < width; ++k) {
        val += M[row * width + k] * N[k * width + col];
      }
      O[row * width + col] = val;
    }
  }
}

__global__ void correctColMatMul(float *M, float *N, float *O, int width) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (col < width) {
    for (int row = 0; row < width; ++row) {
      float val = 0;
      for (int k = 0; k < width; ++k) {
        val += M[row * width + k] * N[k * width + col];
      }
      O[row * width + col] = val;
    }
  }
}

__global__ void vecMatMul(float *B, float *c, float *A, int size) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < size) {
    float val = 0;
    for (int k = 0; k < size; k++) {
      val += B[row * size + k] * c[k];
    }
    A[row] = val;
  }
}

__global__ void expMatMul(float *M, float *N, float *O, int width) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  // float vals[width][width];
  if (row < width && col < width) {
    for (int i = 0; i < width; i++) {
      if (row == i) {
      }
    }
  }
}

void initializeMatrices(float *h_M, float *h_N, int width) {
  for (int i = 0; i < width * width; ++i) {
    h_M[i] = static_cast<float>(i % 10);
    h_N[i] = static_cast<float>(i % 7);
  }
}

void vecMatMul(float *h_B, float *h_c, float *h_A, int size) {
  float *d_B, *d_c, *d_A;
  const int THREADS_PER_BLOCK = 256;
  dim3 dimBlock(1, THREADS_PER_BLOCK, 1);
  const int BLOCKS_PER_GRID =
      (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  dim3 dimGrid(1, BLOCKS_PER_GRID, 1);

  CUDA_CHECK(cudaMalloc(&d_B, (size * size) * sizeof(float)))
  CUDA_CHECK(cudaMalloc(&d_c, size * sizeof(float)))

  CUDA_CHECK(cudaMemcpy(d_B, h_B, (size * size) * sizeof(float),
                        cudaMemcpyHostToDevice))
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice))

  vecMatMul<<<dimGrid, dimBlock>>>(d_B, d_c, d_A, size);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
}

bool compareMatrices(const float *A, const float *B, int width,
                     float epsilon = 1e-5f) {
  for (int i = 0; i < width * width; ++i) {
    if (fabs(A[i] - B[i]) > epsilon) {
      std::cerr << "Matrices differ at index " << i << "! A[" << i
                << "] = " << A[i] << ", B[" << i << "] = " << B[i] << std::endl;
      return false; // Found a difference
    }
  }
  return true; // Matrices are the same
}

int main() {
  const int WIDTH = 1024;
  const int MATRIX_SIZE = WIDTH * WIDTH;
  const size_t BYTES = MATRIX_SIZE * sizeof(float);

  float *h_M = new float[MATRIX_SIZE];
  float *h_N = new float[MATRIX_SIZE];
  float *h_O_row = new float[MATRIX_SIZE];
  float *h_O_correct = new float[MATRIX_SIZE];

  initializeMatrices(h_M, h_N, WIDTH);

  float *d_M, *d_N, *d_O_row, *d_O_correct;
  CUDA_CHECK(cudaMalloc(&d_M, BYTES));
  CUDA_CHECK(cudaMalloc(&d_N, BYTES));
  CUDA_CHECK(cudaMalloc(&d_O_row, BYTES));
  CUDA_CHECK(cudaMalloc(&d_O_correct, BYTES));

  CUDA_CHECK(cudaMemcpy(d_M, h_M, BYTES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_N, h_N, BYTES, cudaMemcpyHostToDevice));

  const int THREADS_PER_BLOCK = 256;
  dim3 dimBlockRow(1, THREADS_PER_BLOCK);
  dim3 dimBlockCol(THREADS_PER_BLOCK, 1);

  const int BLOCKS_PER_GRID =
      (WIDTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  dim3 dimGridRow(1, BLOCKS_PER_GRID);
  dim3 dimGridCol(BLOCKS_PER_GRID, 1);

  std::cout << "Launching Row kernel with grid size (" << dimGridRow.x << ", "
            << dimGridRow.y << ") and block size (" << dimBlockRow.x << ", "
            << dimBlockRow.y << ")" << std::endl;

  std::cout << "Launching Col kernel with grid size (" << dimGridCol.x << ", "
            << dimGridCol.y << ") and block size (" << dimBlockCol.x << ", "
            << dimBlockCol.y << ")" << std::endl;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Run rowMatMul
  std::cout << "Running rowMatMul..." << std::endl;
  CUDA_CHECK(cudaEventRecord(start));
  correctColMatMul<<<dimGridCol, dimBlockCol>>>(d_M, d_N, d_O_row, WIDTH);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

  CUDA_CHECK(cudaMemcpy(h_O_row, d_O_row, BYTES, cudaMemcpyDeviceToHost));

  // Run correctRowMatMul
  std::cout << "Running correctRowMatMul..." << std::endl;
  CUDA_CHECK(cudaEventRecord(start));
  correctRowMatMul<<<dimGridRow, dimBlockRow>>>(d_M, d_N, d_O_correct, WIDTH);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(
      cudaMemcpy(h_O_correct, d_O_correct, BYTES, cudaMemcpyDeviceToHost));

  // Compare the results
  if (compareMatrices(h_O_row, h_O_correct, WIDTH)) {
    std::cout << "The outputs of rowMatMul and correctRowMatMul are the same."
              << std::endl;
  } else {
    std::cout << "The outputs of rowMatMul and correctRowMatMul are "
                 "different."
              << std::endl;
  }

  CUDA_CHECK(cudaFree(d_M));
  CUDA_CHECK(cudaFree(d_N));
  CUDA_CHECK(cudaFree(d_O_row));
  CUDA_CHECK(cudaFree(d_O_correct));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  delete[] h_M;
  delete[] h_N;
  delete[] h_O_row;
  delete[] h_O_correct;

  std::cout << "\nMatrix multiplication complete and resources freed."
            << std::endl;

  return 0;
}
