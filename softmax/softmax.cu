#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <torch/extension.h>

template <typename T>
__global__ void safe_softmax(T *x, T *y, int M, int N) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < M) {
    T x_max = -INFINITY;
    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      x_max = max(x_max, x[i]);
    }

    float d = 0.0f;
    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      d += expf(static_cast<float>(x[i] - x_max));
    }

    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      float n = expf(static_cast<float>(x[i] - x_max));
      y[i] = static_cast<T>(n / d);
    }
  }
}

template <typename T>
__global__ void online_softmax(T *x, T *y, int M, int N) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < M) {
    T x_max = -INFINITY;
    float d = 0.0f;
    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      if (x[i] > x_max) {
        d = d * expf(static_cast<float>(x_max - x[i]));
        x_max = static_cast<float>(x[i]);
      }
      d += expf(static_cast<float>(x[i] - x_max));
    }

    for (int ele = 0; ele < N; ele++) {
      int i = row * N + ele;
      y[i] = static_cast<T>(expf(static_cast<float>(x[i] - x_max)) / d);
    }
  }
}

// final boss: reduction softmax 

struct SoftmaxState {
  float m;
  float d;
};

__device__ SoftmaxState reduceOp(SoftmaxState a, SoftmaxState b) {
  // so we caluclate max of 2 states and cumm_denominator
  // we just multiply 2 d's

  SoftmaxState res;
  res.m = fmaxf(a.m, b.m);

  float factor_a = (a.m == -INFINITY) ? 0.0f : __expf(a.m - res.m);
  float factor_b = (b.m == -INFINITY) ? 0.0f : __expf(b.m - res.m);

  res.d = a.d * factor_a + b.d * factor_b;
  return res;
}

__device__ SoftmaxState warpReduceSoftmax(SoftmaxState val) {
  // so basically we find max and d for all the elements in warp 
  for (int offset = 16; offset > 0; offset/=2) {
    // fetching values from offset threads away (i.e. 16, 8, 4, 2)
    // its like dark magic
    float other_m = __shfl_down_sync(0xffffffff, val.m, offset);
    float other_d = __shfl_down_sync(0xffffffff, val.d, offset);

    SoftmaxState other = {other_m, other_d};

    // merging states by get max of both and cumm_d
    val = reduceOp(val, other);
  }
  return val;
}

template <typename T>
__global__ void reduceSoftmaxKernel(T *input, T *output, int M, int N) {
  int row = blockIdx.x;

  if (row >= M) return;

  // adjust pointers to point to the start of the specific row
  T *row_input = input + row*N;
  T *row_output = output + row*N;

  // this will store warp-level states 
  // assuming max 32 warps (1024 threads)
  __shared__ float shared_m[32];
  __shared__ float shared_d[32];

  int tid = threadIdx.x;
  int laneId = tid % 32;
  int warpId = tid / 32;

  // warp level local state
  SoftmaxState localState = {-INFINITY, 0.0f};

  // strided loop
  // if N > blockDim.x 
  for (int i = tid; i < N; i+=blockDim.x) {
    float val = static_cast<float>(row_input[i]);

    // standard online softmax updates 
    float new_m = fmaxf(localState.m, val);
    float factor = __expf(localState.m - new_m);
    localState.d = localState.d * factor + __expf(val - new_m);
    localState.m = new_m;
  }

  // warp-level reduction 
  localState = warpReduceSoftmax(localState);

  if (laneId == 0) {
    shared_m[warpId] = localState.m;
    shared_d[warpId] = localState.d;
  }
  __syncthreads();

  // now we will reduce all of the localState's of different warps in warp[0]
  if (warpId == 0) {
    SoftmaxState warpState = (tid < (blockDim.x / 32)) ?
                              SoftmaxState{shared_m[tid], shared_d[tid]} :
                              SoftmaxState{-INFINITY, 0.0f};

    warpState = warpReduceSoftmax(warpState);

    if (tid == 0) {
      shared_m[0] = warpState.m;
      shared_d[0] = warpState.d;
    }
  }
  __syncthreads();

  float total_m = shared_m[0];
  float total_d = shared_d[0];

  // final normed softmax pass
  for (int i = tid; i < N; i += blockDim.x) {
    float val = static_cast<float>(row_input[i]);
    row_output[i] = static_cast<T>(__expf(val - total_m) / total_d);
  }
}

torch::Tensor dispatch_safe_softmax(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
  input = input.contiguous();
  int M = input.size(0);
  int N = input.size(1);
  auto output = torch::empty_like(input);

  // Naive configuration: 1 thread per row
  int threads = 128;
  int blocks = (M + threads - 1) / threads;

 AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "safe_softmax", ([&] {
      safe_softmax<scalar_t><<<blocks, threads>>>(
          input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), M, N);
    }));
  return output;
}

torch::Tensor dispatch_online_softmax(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
  input = input.contiguous();
  int M = input.size(0);
  int N = input.size(1);
  auto output = torch::empty_like(input);

  // Naive configuration: 1 thread per row
  int threads = 128;
  int blocks = (M + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "online_softmax", ([&] {
    online_softmax<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), M, N);
  }));
  return output;
}

torch::Tensor dispatch_reduce_softmax(torch::Tensor input) {
  // ensure inputs are on the GPU
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  
  // ensure the memory is contiguous (Row-Major)
  input = input.contiguous();

  int M = input.size(0);
  int N = input.size(1);
  
  auto output = torch::empty_like(input);

  // Kernel Launch Configuration
  // 1 Block per Row, 1024 Threads per Block
  dim3 grid(M);
  dim3 block(1024); 

  // The Magic Macro: Automatically handles fp32, fp16, bf16
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "reduceSoftmaxKernel", ([&] {
    reduceSoftmaxKernel<scalar_t><<<grid, block>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        M,
        N);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("safe_softmax", &dispatch_safe_softmax, "Naive 3-pass Softmax");
  m.def("online_softmax", &dispatch_online_softmax, "Online 2-pass Softmax");
  m.def("reduce_softmax", &dispatch_reduce_softmax, "Optimized Warp Reduction Softmax");
}
