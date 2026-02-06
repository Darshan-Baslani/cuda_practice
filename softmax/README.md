# Softmax Optimization Journey

From numerically stable baselines to warp-level parallel reductions.

This project explores optimizing the softmax operation — a critical component in attention mechanisms and neural network classifiers. The journey covers numerical stability, algorithmic improvements, and GPU-native parallel reduction techniques.

---

## Why Softmax is Tricky

The naive formula `softmax(x)_i = exp(x_i) / Σ exp(x_j)` has a fatal flaw:

```
exp(1000) = Infinity    // Overflow!
exp(-1000) = 0          // Underflow, then 0/0 = NaN
```

In deep learning, logits routinely reach values of 100+. A production softmax must handle this gracefully.

---

## The Journey: 3 Kernels, 3 Paradigms

### Kernel 1: Safe Softmax (3-Pass)
**Correctness first** — the numerically stable baseline.

```cuda
__global__ void safe_softmax(T *x, T *y, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    // Pass 1: Find maximum (for numerical stability)
    T x_max = -INFINITY;
    for (int ele = 0; ele < N; ele++) {
        x_max = max(x_max, x[row * N + ele]);
    }

    // Pass 2: Compute denominator with shifted values
    float d = 0.0f;
    for (int ele = 0; ele < N; ele++) {
        d += expf(x[row * N + ele] - x_max);  // Now safe!
    }

    // Pass 3: Normalize
    for (int ele = 0; ele < N; ele++) {
        float n = expf(x[row * N + ele] - x_max);
        y[row * N + ele] = n / d;
    }
}
```

**What I learned**:
- Subtracting the max before `exp()` keeps values in a safe range
- `exp(x - max)` is mathematically equivalent but numerically stable
- Three passes = three full reads of the row from global memory
- This is how PyTorch's softmax works internally (with optimizations)

---

### Kernel 2: Online Softmax (2-Pass)
**Algorithmic improvement** — computing max and sum in one pass.

```cuda
__global__ void online_softmax(T *x, T *y, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    T x_max = -INFINITY;
    float d = 0.0f;

    // Pass 1: Online update of max AND denominator
    for (int ele = 0; ele < N; ele++) {
        float val = x[row * N + ele];
        if (val > x_max) {
            // Key insight: rescale running sum when max changes
            d = d * expf(x_max - val);
            x_max = val;
        }
        d += expf(val - x_max);
    }

    // Pass 2: Normalize (still need to read data again)
    for (int ele = 0; ele < N; ele++) {
        y[row * N + ele] = expf(x[row * N + ele] - x_max) / d;
    }
}
```

**What I learned**:
- Welford's online algorithm: update statistics incrementally
- When we find a new max, the old sum must be rescaled: `d *= exp(old_max - new_max)`
- This reduces memory traffic by 33% (2 passes instead of 3)
- Algorithmic improvements often beat micro-optimizations

**The math behind rescaling**:
```
Old sum: d = Σ exp(x_i - old_max)
New sum: d' = Σ exp(x_i - new_max)
       = Σ exp(x_i - old_max + old_max - new_max)
       = Σ exp(x_i - old_max) * exp(old_max - new_max)
       = d * exp(old_max - new_max)
```

---

### Kernel 3: Reduction Softmax (Warp-Level)
**GPU-native parallelism** — the final boss.

The previous kernels use one thread per row. For wide rows (common in transformers with seq_len=2048+), this leaves most of the GPU idle. The solution: parallelize *within* each row.

```cuda
struct SoftmaxState {
    float m;  // max value
    float d;  // denominator (sum of exp)
};

// Combine two partial softmax states
__device__ SoftmaxState reduceOp(SoftmaxState a, SoftmaxState b) {
    SoftmaxState res;
    res.m = fmaxf(a.m, b.m);

    // Rescale both denominators to the new max
    float factor_a = (a.m == -INFINITY) ? 0.0f : __expf(a.m - res.m);
    float factor_b = (b.m == -INFINITY) ? 0.0f : __expf(b.m - res.m);

    res.d = a.d * factor_a + b.d * factor_b;
    return res;
}
```

**Warp-level reduction using shuffle intrinsics**:
```cuda
__device__ SoftmaxState warpReduceSoftmax(SoftmaxState val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        // Threads exchange data within the warp — no shared memory needed!
        float other_m = __shfl_down_sync(0xffffffff, val.m, offset);
        float other_d = __shfl_down_sync(0xffffffff, val.d, offset);

        val = reduceOp(val, {other_m, other_d});
    }
    return val;
}
```

**The full reduction hierarchy**:
```cuda
__global__ void reduceSoftmaxKernel(T *input, T *output, int M, int N) {
    int row = blockIdx.x;  // One block per row
    int tid = threadIdx.x;

    // Step 1: Each thread processes strided elements
    SoftmaxState localState = {-INFINITY, 0.0f};
    for (int i = tid; i < N; i += blockDim.x) {
        float val = input[row * N + i];
        // Online softmax update
        float new_m = fmaxf(localState.m, val);
        localState.d = localState.d * __expf(localState.m - new_m)
                     + __expf(val - new_m);
        localState.m = new_m;
    }

    // Step 2: Warp-level reduction (32 threads → 1 state)
    localState = warpReduceSoftmax(localState);

    // Step 3: Store warp results to shared memory
    __shared__ float shared_m[32], shared_d[32];
    if (laneId == 0) {
        shared_m[warpId] = localState.m;
        shared_d[warpId] = localState.d;
    }
    __syncthreads();

    // Step 4: First warp reduces all warp results
    if (warpId == 0) {
        SoftmaxState warpState = {shared_m[tid], shared_d[tid]};
        warpState = warpReduceSoftmax(warpState);
        if (tid == 0) {
            shared_m[0] = warpState.m;
            shared_d[0] = warpState.d;
        }
    }
    __syncthreads();

    // Step 5: Final normalization pass
    float total_m = shared_m[0], total_d = shared_d[0];
    for (int i = tid; i < N; i += blockDim.x) {
        output[row * N + i] = __expf(input[row * N + i] - total_m) / total_d;
    }
}
```

**What I learned**:
- `__shfl_down_sync` allows threads in a warp to share data without shared memory
- Parallel reduction: O(log n) steps instead of O(n) serial iterations
- The challenge: combining partial softmax states requires careful rescaling
- Block organization: 1 block per row, 1024 threads per block
- Two-level reduction: first within warps (fast), then across warps (shared memory)

---

## Key Insights

### Combining Online Softmax with Parallel Reduction

This is the non-obvious part. Each thread computes a partial `(max, denominator)` pair. But you can't just take `max(all_maxes)` and `sum(all_denoms)` — the denominators were computed with different max values!

The solution: when combining two states, rescale both denominators to the combined max:
```
combined.max = max(a.max, b.max)
combined.denom = a.denom * exp(a.max - combined.max)
               + b.denom * exp(b.max - combined.max)
```

This is what makes `reduceOp` associative, enabling parallel reduction.

### Why Warp Shuffles?

| Method | Latency | Sync Required |
|--------|---------|---------------|
| Global Memory | ~400 cycles | Yes |
| Shared Memory | ~20 cycles | Yes (`__syncthreads`) |
| Warp Shuffle | ~2 cycles | No (implicit in warp) |

Shuffles are 10-200x faster than alternatives for intra-warp communication.

---

## Multi-Precision Support

All kernels support fp32, fp16, and bf16 through PyTorch's dispatch macro:

```cuda
AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "kernel_name", ([&] {
    my_kernel<scalar_t><<<grid, block>>>(...);
}));
```

**What I learned**:
- `scalar_t` is a compile-time type alias resolved by the macro
- Internal computations use fp32 for precision (`static_cast<float>`)
- Inputs/outputs can be lower precision for memory bandwidth
- This is how PyTorch extensions handle mixed precision

---

## PyTorch Integration

Built as a C++ extension using pybind11:

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("safe_softmax", &dispatch_safe_softmax, "Naive 3-pass Softmax");
    m.def("online_softmax", &dispatch_online_softmax, "Online 2-pass Softmax");
    m.def("reduce_softmax", &dispatch_reduce_softmax, "Optimized Warp Reduction");
}
```

**Usage in Python**:
```python
import torch
import softmax_cpp

x = torch.randn(1024, 32768, device='cuda')
y = softmax_cpp.reduce_softmax(x)  # Custom kernel
z = torch.softmax(x, dim=1)        # PyTorch native

assert torch.allclose(y, z, atol=1e-2)
```

---

## Benchmarking

Test configuration: 1024 rows × 32768 columns (float32)

```bash
python benchmark.py
```

The benchmark:
1. Verifies correctness against PyTorch's native softmax
2. Runs 100 iterations with warmup
3. Reports average time and speedup vs PyTorch

---

## Building

Requires: PyTorch with CUDA support

```bash
pip install -e .
```

Or for development:
```bash
python setup.py build_ext --inplace
```

This compiles `softmax.cu` into a Python-importable shared object.

---

## Files

```
softmax/
├── softmax.cu       # All kernel implementations
├── setup.py         # Build configuration
├── benchmark.py     # Performance testing
└── README.md        # This file
```

---

## What's Next

- [ ] Flash Attention-style fused softmax (online softmax + matmul)
- [ ] Multi-block reduction for extremely wide rows
- [ ] Memory-bound analysis with Nsight Compute
- [ ] Comparison with Triton implementation
