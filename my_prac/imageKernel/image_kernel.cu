#include <cuda_runtime.h>
#include <image_kernel.cuh>

__global__ void rgbToGrayscale(unsigned char *pin, unsigned char *pout,
                               int width, int height, int channels) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * channels;

    unsigned char r = pin[rgbOffset];
    unsigned char g = pin[rgbOffset + 1];
    unsigned char b = pin[rgbOffset + 2];

    pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

void callRgbToGrayscaleKernel(unsigned char *pin, unsigned char *pout, int width,
                              int height, int channels) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  rgbToGrayscale<<<numBlocks, threadsPerBlock>>>(pin, pout, width, height,
                                                  channels);
}
