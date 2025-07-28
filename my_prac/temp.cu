#include <cstdio> // Required for printf
#include <cuda_runtime.h>
#include <iostream>

__global__ void globalThreadVisualizer() {
  // Using printf for device-side output
  printf("X axis : %d\n Y axis : %d\n", (threadIdx.x + blockDim.x * blockIdx.x),
         (threadIdx.y + blockDim.y * blockIdx.y));
}

int main() {
  dim3 nBlocks(2, 2);
  dim3 nThreads(2, 2);
  globalThreadVisualizer<<<nBlocks, nThreads>>>();

  std::cout << "Kernel launched successfully." << std::endl;

  cudaDeviceSynchronize();

  return 0;
}
