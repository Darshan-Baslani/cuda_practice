#include <cuda_runtime.h>
#include <image_kernel.cuh>
#include <iostream>
#include <opencv2/opencv.hpp>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err = x;                                                       \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s %s:%d: %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main() {
  // 1. Load the image on the CPU using OpenCV
  cv::Mat image = cv::imread("image_kernel/images/ding.jpg", cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return -1;
  }

  // Get raw pixel data pointer from OpenCV Mat
  unsigned char *h_input_data = image.data;
  int width = image.cols;
  int height = image.rows;
  int channels = image.channels();

  // Allocate host memory for the grayscale output image
  unsigned char *h_output_data = new unsigned char[width * height];

  // 2. Allocate device memory and copy host data to device
  unsigned char *d_input_data;
  unsigned char *d_output_data;

  CUDA_CHECK(cudaMalloc((void **)&d_input_data,
                         width * height * channels * sizeof(unsigned char)));
  CUDA_CHECK(cudaMalloc((void **)&d_output_data,
                         width * height * sizeof(unsigned char)));

  CUDA_CHECK(cudaMemcpy(d_input_data, h_input_data,
                        width * height * channels * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));

  // 3. Launch the kernel
  callRgbToGrayscaleKernel(d_input_data, d_output_data, width, height, channels);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 4. Copy the result back to the host
  CUDA_CHECK(cudaMemcpy(h_output_data, d_output_data,
                        width * height * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));

  // 5. Save the grayscale image
  cv::Mat grayscale_image(height, width, CV_8UC1, h_output_data);
  cv::imwrite("images/ding_grayscale.jpg", grayscale_image);
  std::cout << "Grayscale image saved to images/ding_grayscale.jpg" << std::endl;

  // Free device memory
  cudaFree(d_input_data);
  cudaFree(d_output_data);

  // Free host memory
  delete[] h_output_data;

  return 0;
}
