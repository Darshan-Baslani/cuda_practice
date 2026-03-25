import torch
import triton
import triton.testing

# import softmax_cpp  <-- Uncomment this to use your actual compiled extension

# Keep rows fixed for this test, but you could also make this a variable!
ROWS = 1024

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['cols'],  
        x_vals=[1024 * 2**i for i in range(1, 7)],  
        x_log=True,        
        line_arg='provider', 
        line_vals=['torch', 'safe', 'online', 'reduce'], 
        line_names=['PyTorch Native', 'Safe (Naive)', 'Online (Naive)', 'Reduce (Optimized)'], 
        styles=[('blue', '-'), ('red', '--'), ('orange', '--'), ('green', '-')], 
        ylabel='Execution Time (ms)', 
        plot_name='softmax-performance', 
        args={},  
    )
)
def benchmark(cols, provider):
    x = torch.randn(ROWS, cols, device='cuda', dtype=torch.float32)

    # We MUST define the quantiles we want to get the tuple back
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
    elif provider == 'safe':
        # Replace with: lambda: softmax_cpp.safe_softmax(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
    elif provider == 'online':
        # Replace with: lambda: softmax_cpp.online_softmax(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
    elif provider == 'reduce':
        # Replace with: lambda: softmax_cpp.reduce_softmax(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)

    return ms, max_ms, min_ms

if __name__ == "__main__":
    print(f"Benchmarking Softmax (Fixed Rows: {ROWS}) across various Column sizes...")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("-" * 60)
    
    # Run the benchmark
    # show_plots=True: Opens a window with the graph (if in a GUI environment)
    # print_data=True: Prints a neat markdown table to the console
    # save_path: Saves the raw CSV data and a PNG graph to this folder
    benchmark.run(print_data=True, show_plots=False, save_path='./bench_results')
