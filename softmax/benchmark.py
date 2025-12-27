import torch
import softmax_cpp
import time
import pandas as pd

device = torch.device("cuda")
rows, cols = 1024, 32768
print(f"Benchmarking Softmax on {rows}x{cols} matrix (float32")
print(f"Device: {torch.cuda.get_device_name(0)}")
print("-" * 60)

x = torch.randn(rows, cols, device=device, dtype=torch.float32)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    torch_out = torch.softmax(x, dim=1)
end.record()
torch.cuda.synchronize()
torch_time = start.elapsed_time(end) / 100
print(f"PyTorch Native: {torch_time:.4f} ms")

kernels = [
    ("Safe (Naive)", softmax_cpp.safe_softmax),
    ("Online (Naive)", softmax_cpp.online_softmax),
    ("Reduce (Optimized)", softmax_cpp.reduce_softmax)
]

results = []

for name, kernel_func in kernels:
    print(f"\nTesting {name}...")
    
    # Correctness Check
    try:
        my_out = kernel_func(x)
        diff = torch.abs(torch_out - my_out).max().item()
        status = "PASS" if diff < 1e-2 else f"FAIL (Diff: {diff:.5f})"
        print(f"  Logic Check: {status}")
    except Exception as e:
        print(f"  Logic Check: CRASHED - {e}")
        continue

    # --- Benchmark ---
    # Warmup
    for _ in range(10): kernel_func(x)
    
    start.record()
    for _ in range(100):
        kernel_func(x)
    end.record()
    torch.cuda.synchronize()
    
    avg_time = start.elapsed_time(end) / 100
    print(f"  Avg Time:    {avg_time:.4f} ms")
    
    results.append({
        "Kernel": name,
        "Time (ms)": avg_time,
        "Speedup vs Torch": torch_time / avg_time
    })

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
df = pd.DataFrame(results)
print(df.to_string(index=False))
