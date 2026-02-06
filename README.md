# CUDA Optimization Journey

A hands-on exploration of GPU programming, from first principles to beating cuBLAS.

This repository documents my journey learning CUDA through implementing progressively optimized matrix multiplication (GEMM) and softmax kernels. Each kernel builds on lessons from the previous, demonstrating a deepening understanding of GPU architecture.

**Key Achievement**: My 2D Block Tiling kernel achieves **110% of cuBLAS performance** on matrix multiplication.

![Benchmark Results](gemm/benchmark_results.png)

---

## The GEMM Journey: 7 Kernels, 7 Lessons

### Kernel 1: Naive Implementation
**The baseline** — understanding how GPUs execute parallel work.

```cuda
// Each thread computes one output element
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
}
C[row * N + col] = sum;
```

**What I learned**:
- CUDA's thread hierarchy (grids, blocks, threads)
- How to map 2D matrix operations to thread indices
- Why this is slow: every operation hits global memory (~400 cycles latency)

---

### Kernel 2: Global Memory Coalescing
**First optimization** — working with the hardware, not against it.

**What I learned**:
- Adjacent threads should access adjacent memory locations
- Memory transactions happen in 32/128-byte chunks
- Poor access patterns waste bandwidth fetching unused data
- Arranging threads to match row-major layout eliminates wasted transactions

---

### Kernel 3: Shared Memory Tiling
**The big leap** — exploiting the memory hierarchy.

```cuda
__shared__ float As[TILESIZE][TILESIZE];
__shared__ float Bs[TILESIZE][TILESIZE];

// Load tile cooperatively
As[threadIdx.y][threadIdx.x] = A[...];
Bs[threadIdx.y][threadIdx.x] = B[...];
__syncthreads();

// Compute using fast shared memory
for (int k = 0; k < TILESIZE; k++) {
    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
}
```

**What I learned**:
- Shared memory is ~100x faster than global memory
- Tiling: load a block of data once, reuse it many times
- `__syncthreads()` ensures all threads finish loading before computing
- The importance of tile size tuning for occupancy

---

### Kernel 4: 1D Block Tiling
**More work per thread** — improving arithmetic intensity.

**What I learned**:
- Having each thread compute multiple outputs (TM=8) amortizes load costs
- Arithmetic intensity = compute ops / memory ops (higher is better)
- 1D thread organization simplifies indexing while maintaining efficiency
- Register pressure becomes a consideration

---

### Kernel 5: 2D Block Tiling
**Register-level optimization** — where the magic happens.

```cuda
float regA[TM];  // Cache A values in registers
float regB[TN];  // Cache B values in registers
float result[TM * TN] = {0};  // 64 outputs per thread

// Load from shared memory to registers
for (int i = 0; i < TM; i++)
    regA[i] = As[(threadRow * TM + i) * BK + k];
for (int i = 0; i < TN; i++)
    regB[i] = Bs[k * BN + threadCol * TN + i];

// Compute in registers (fastest memory)
for (int i = 0; i < TM; i++)
    for (int j = 0; j < TN; j++)
        result[i * TN + j] += regA[i] * regB[j];
```

**What I learned**:
- Registers are the fastest memory (~0 cycle latency)
- Each thread computing a TM×TN=8×8 tile maximizes register reuse
- Block dimensions (BM=128, BN=128, BK=8) tuned for my GPU (GTX 1650)
- This is where I surpassed cuBLAS — understanding *why* was enlightening

---

### Kernel 6: Vectorized Loads
**Maximizing memory bandwidth** — doing more with each transaction.

```cuda
// Load 4 floats at once using float4
float4 tmp = reinterpret_cast<float4*>(&A[...])[0];
As[k * BM + innerRowA * 4 + 0] = tmp.x;
As[k * BM + innerRowA * 4 + 1] = tmp.y;
As[k * BM + innerRowA * 4 + 2] = tmp.z;
As[k * BM + innerRowA * 4 + 3] = tmp.w;

// Padding to avoid bank conflicts
__shared__ float Bs[BK * (BN + 5)];  // +5 padding
```

**What I learned**:
- `float4` loads issue one instruction for 128 bits instead of four
- Memory layout affects shared memory bank conflicts (32 banks, 4-byte words)
- Padding shared memory arrays can eliminate bank conflicts entirely
- Transposing data during loads can improve subsequent access patterns

---

### Kernel 7: Warp Tiling
**Hierarchical optimization** — thinking at every level.

```
Block Tile (128×128)
  └── Warp Tiles (64×64)
        └── Sub-warp Tiles (WSUBM×WSUBN)
              └── Thread Tiles (4×4)
```

**What I learned**:
- Warps (32 threads) execute in lockstep — they're the true unit of execution
- Warp-level tiling reduces synchronization overhead
- Hierarchical blocking: optimize data movement at block, warp, and thread levels
- The tradeoff between occupancy and per-thread resources

---

## The Softmax Journey: Numerical Stability Meets Performance

### Kernel 1: Safe Softmax (3-Pass)
**Correctness first** — understanding why naive softmax breaks.

```
Pass 1: max_val = max(x)           // Prevent overflow
Pass 2: sum = Σ exp(x - max_val)   // Stable computation
Pass 3: output = exp(x - max_val) / sum
```

**What I learned**:
- `exp(large_number)` overflows to infinity
- Subtracting the maximum keeps values in a safe range
- Numerical stability is non-negotiable in production code

---

### Kernel 2: Online Softmax (2-Pass)
**Algorithmic improvement** — fewer passes, same stability.

**What I learned**:
- Welford's online algorithm: update statistics incrementally
- When a new max is found, rescale the running sum: `sum *= exp(old_max - new_max)`
- Reducing memory passes is often more impactful than micro-optimizations

---

### Kernel 3: Reduction Softmax (Warp-Level)
**GPU-native algorithms** — using hardware primitives.

```cuda
// Warp-level reduction using shuffle
for (int offset = 16; offset > 0; offset /= 2) {
    float other_max = __shfl_down_sync(0xffffffff, state.max, offset);
    float other_denom = __shfl_down_sync(0xffffffff, state.denom, offset);
    // Combine states...
}
```

**What I learned**:
- `__shfl_down_sync` allows threads in a warp to share data without shared memory
- Parallel reduction: O(log n) steps instead of O(n)
- Combining online softmax with parallel reduction is non-trivial but powerful
- Multi-precision support (fp32, fp16, bf16) via PyTorch's dispatch macros

---

## Performance Results

At 4096×4096 matrix multiplication:

| Kernel | GFLOPS | % of cuBLAS |
|--------|--------|-------------|
| cuBLAS (baseline) | 1,798.9 | 100% |
| 2D Block Tiling | 1,989.6 | **110.6%** |

Peak performance of 2,009 GFLOPS achieved at 2048×2048.

---

## Technical Stack

- **Language**: CUDA C++
- **Build System**: CMake
- **Verification**: Comparison against cuBLAS for correctness
- **Benchmarking**: Custom timing harness with warm-up runs
- **Python Integration**: PyTorch C++ extensions (pybind11) for softmax kernels

---

## Key Takeaways

1. **Memory is everything**: The GPU memory hierarchy (registers → shared → global) dominates performance
2. **Arithmetic intensity matters**: More compute per byte loaded = better utilization
3. **Work with the hardware**: Coalescing, bank conflicts, warp execution — understanding these is essential
4. **Measure, don't guess**: Profiling revealed surprises at every optimization step
5. **Algorithms matter too**: Online softmax shows that clever algorithms complement hardware optimization

---

## Building

```bash
cd gemm
mkdir build && cd build
cmake ..
make
./gemm  # Runs benchmarks
```

For softmax (requires PyTorch):
```bash
cd softmax
pip install -e .
python benchmark.py
```

---

## Resources That Helped

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Simon Boehm's GEMM Tutorial](https://siboehm.com/articles/22/CUDA-MMM)
- [Lei Mao's Blog](https://leimao.github.io/)
- NVIDIA's cuBLAS source (for understanding what "good" looks like)
