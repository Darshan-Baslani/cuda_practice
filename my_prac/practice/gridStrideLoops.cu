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

__global__ void gridStrideLoops(int n, float s, float *M, float *N) {
  for (int i = blockIdx.x*blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x*gridDim.x) {
      N[i] = s * M[i] + N[i];
  }
}

void initializeArray(int n, float *M, float *N) {
  for (int i = 0; i<n; i++) {
    M[i] = static_cast<float>(i % 10);
    N[i] = static_cast<float>(i % 7);
  }
}

int main() {
  int n = 1 << 20; // 1 million
  float *M = new float[n];
  float *N = new float[n];
  float *Md, *Nd;

  initializeArray(n, M, N);

  CUDA_CHECK(cudaMalloc(&Md, (n * sizeof(float))));
  CUDA_CHECK(cudaMalloc(&Nd, (n * sizeof(float))));

  CUDA_CHECK(cudaMemcpy(Md, M, n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Nd, N, n, cudaMemcpyHostToDevice));

  int numSMs;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

  gridStrideLoops<<<32*numSMs, 256>>>(n, 2.9, Md, Nd);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  return 0;
}
