#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(res) {                                                      \
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

__global__ void coalesced_access(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Each thread accesses consecutive 4-byte words
        output[tid] = input[tid] * 2.0f ;
    }
}

__global__ void uncoalesced_access(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Access with a stride of 32 (128 bytes), wrapped around to stay within bounds
        int scattered_index = (tid * 32) % n;
        output[tid] = input[scattered_index] * 2.0f;
    }
}

void launch_coalesced_kernel(float *input, float *output, int n) {
  float *inputD, *outputD;
  CUDA_CHECK(cudaMalloc(&inputD, (n * sizeof(float))));
  CUDA_CHECK(cudaMalloc(&outputD, (n * sizeof(float))));

  CUDA_CHECK(cudaMemcpy(inputD, input, n * sizeof(float), cudaMemcpyHostToDevice));

  int blockDim = 128;
  int gridDim = (n + blockDim - 1) / blockDim;

  std::cout << "launching coalesced kernel" << std::endl;

  coalesced_access<<<gridDim, blockDim>>>(inputD, outputD, n);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "kernel exectuted successfully" << std::endl;
}

void launch_uncoalesced_kernel(float *input, float *output, int n) {
  float *inputD, *outputD;
  CUDA_CHECK(cudaMalloc(&inputD, (n * sizeof(float))));
  CUDA_CHECK(cudaMalloc(&outputD, (n * sizeof(float))));

  CUDA_CHECK(cudaMemcpy(inputD, input, n * sizeof(float), cudaMemcpyHostToDevice));

  int blockDim = 128;
  int gridDim = (n + blockDim - 1) / blockDim;

  std::cout << "launching uncoalesced kernel" << std::endl;

  uncoalesced_access<<<gridDim, blockDim>>>(inputD, outputD, n);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "kernel exectuted successfully" << std::endl;
}

void initialize_vector(float *input, int n) {
  for (int i = 0; i < n; i++) {
    input[i] = static_cast<float>(i % 10);
  }
}

int main() {
  int n = 10000000;
  float *input = new float[n];
  float *output = new float[n];

  initialize_vector(input, n);

  launch_coalesced_kernel(input, output, n);
  launch_uncoalesced_kernel(input, output, n);

  return 0;
}
