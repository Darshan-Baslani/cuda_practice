// given a matrix m; we will add 1 to each element in the matrix

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstdint>
#include <cstdlib>
#include <cuda.h> // CUtensorMap
#include <cuda/barrier>
#include <cuda_runtime.h>

#include "utils.h"

#define CUDA_CHECK(res)                                                        \
  {                                                                            \
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

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *file, int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <int BLOCK_SIZE>
__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) {
  // aligning shared mem to 128 byte; we could use alginas(128/256/...)
  // but 1024 works good across different archs
  __shared__ alignas(1024) int smem_buffer[BLOCK_SIZE * BLOCK_SIZE];

  int x = blockIdx.x * BLOCK_SIZE;
  int y = blockIdx.y * BLOCK_SIZE;

  int col = threadIdx.x % BLOCK_SIZE;
  int row = threadIdx.x / BLOCK_SIZE;

#pragma nv_diag_supress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    // make initialized barrier visible in async proxy.
    cuda::device::experimental::fence_proxy_async_shared_cta();
  }
  // sync to make bar visible to all threads
  __syncthreads();

  cuda::barrier<cuda::thread_scope_block>::arival_token token;
  if (threadIdx.x == 0) {
    // initializing bulk tensor copy
    cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
        &smem_bufferm, &tensor_map, x, y, bar);
    // arrive on the barrier and tell how many bytes are expected to come in
    token = cuda::device:barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // other threads just arrive 
    token = bar.arrive;
  }

  // wait for the data to have arrived 
  bar.wait(std::move(token));

  // adding 1
  smem_buffer[row*BLOCK_SIZE + col] += 1;
  
  // wait for the shared mem writes to be visible to TMA engine 
  cuda::device::experimental::fence_proxy_async_shared_cta();
  __syncthreads();

  // TMA shared -> global
  if (threadIdx.x == 0) {
    cuda::device::experimental::cp_async_bulk_tensor_2d_shared_to_global(
        &tensor_map, x, y, &smem_buffer
      );

    // what??
    cuda::device::experimental::cp_async_bulk_commit_group();
    //wait for the group to have completed reading from shared mem 
    cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0) {
    (&bar)->~cuda::barrier<cuda::thread_scope_block>();
  }
}

int main() {
  const uint GMEM_WIDTH = 8192;
  const uint GMEM_HEIGHT = 8192;
  const uint BLOCK_SIZE = 32;
  const uint SMEM_WIDTH = BLOCK_SIZE;
  const uint SMEM_HEIGHT = BLOCK_SIZE;
  const size_t SIZE = GMEM_HEIGHT * GMEM_WIDTH * sizeof(float);

  float *h_in = new float[GMEM_HEIGHT * GMEM_WIDTH];
  float *h_out = new float[GMEM_HEIGHT * GMEM_WIDTH];

  srand(369);
  for (uint i{0}; i < GMEM_HEIGHT * GMEM_WIDTH; i++) {
    h_in[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  float *device_ptr;
  CUDA_CHECK(cudaMalloc(&device_ptr, SIZE));
  CUDA_CHECK(cudaMemcpy(device_ptr, h_in, SIZE, cudaMemcpyHostToDevice));
  void *tensor_ptr = (void *)device_ptr;

  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_HEIGHT, GMEM_WIDTH};
  // row major stride
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(float)};
  // box size is the size of shared mem
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  // distance between elements in units of sizeof(elements)
  uint32_t elem_stride[rank] = {1, 1};

  // creating the tensor descriptor
  CUresult res = cuTensorMapEncodeTiled(
      &tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32, rank,
      tensor_ptr, size, stride, box_size, ele,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(res == CUDA_SUCCESS);

  dim3 blockDim(SMEM_WIDTH * SMEM_HEIGHT, 1, 1);
  dim3 gridDim(GMEM_WIDTH / SMEM_WIDTH, GMEM_HEIGHT / SMEM_HEIGHT, 1);

  kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(tensor_map);

  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaMemcpy(h_out, d, SIZE, cudaMemcpyDeviceToHost));

  // std::cout << "Matrix after launching kernel:" << std::endl;
  // utils::printMatrix(h_out, GMEM_HEIGHT, GMEM_WIDTH);
  // std::cout << std::endl;

  for (int x = 0; x < GMEM_HEIGHT; x++) {
    for (int y = 0; y < GMEM_WIDTH; y++) {
      if (h_out[x * GMEM_WIDTH + y] != h_in[x * GMEM_WIDTH + y] + 1) {
        std::cout << "Error at position (" << x << "," << y << "): expected "
                  << h_in[x * GMEM_WIDTH + y] + 1 << " but got "
                  << h_out[x * GMEM_WIDTH + y] << std::endl;
        return -1;
      }
    }
  }
  std::cout << "Passed" << std::endl;

  CHECK_CUDA_ERROR(cudaFree(d));
  free(h_in);
  free(h_out);

  return 0;
}
