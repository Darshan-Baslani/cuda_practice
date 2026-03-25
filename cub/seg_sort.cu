#include <cub/cub.cuh>
#include <iostream>
#include <vector>

// Helper macro for checking CUDA errors
#define cudaCheck(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(err); \
    } \
}

int main() {
    // 1. Initialize host data
    // We have 10 items total, divided into 3 segments:
    // Segment 0: [8, 6, 7]       (indices 0-2)
    // Segment 1: [5, 3, 0, 9]    (indices 3-6)
    // Segment 2: [2, 4, 1]       (indices 7-9)
    int num_items = 10;
    int num_segments = 3;
    
    std::vector<int> h_keys_in = {8, 6, 7, 5, 3, 0, 9, 2, 4, 1};
    // Offsets define the start of each segment, plus a final offset for the end
    std::vector<int> h_offsets = {0, 3, 7, 10}; 
    std::vector<int> h_keys_out(num_items);

    // 2. Allocate device memory
    int *d_keys_in = nullptr;
    int *d_keys_out = nullptr;
    int *d_offsets = nullptr;

    cudaCheck(cudaMalloc(&d_keys_in, num_items * sizeof(int)));
    cudaCheck(cudaMalloc(&d_keys_out, num_items * sizeof(int)));
    cudaCheck(cudaMalloc(&d_offsets, (num_segments + 1) * sizeof(int)));

    // 3. Copy data from host to device
    cudaCheck(cudaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_offsets, h_offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // 4. Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // First call with d_temp_storage = nullptr to get the required temp storage size
    cudaCheck(cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1)); // d_offsets is begin_offsets, d_offsets + 1 is end_offsets

    // 5. Allocate temporary storage
    cudaCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // 6. Perform the actual segmented sort
    cudaCheck(cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1));

    // 7. Copy results back to host
    cudaCheck(cudaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost));

    // 8. Print the results
    std::cout << "Original Array: ";
    for (int i = 0; i < num_items; i++) std::cout << h_keys_in[i] << " ";
    std::cout << "\n\n";

    std::cout << "Sorted Array (by segment):\n";
    for (int seg = 0; seg < num_segments; seg++) {
        std::cout << "Segment " << seg << ": ";
        for (int i = h_offsets[seg]; i < h_offsets[seg+1]; i++) {
            std::cout << h_keys_out[i] << " ";
        }
        std::cout << "\n";
    }

    // 9. Clean up device memory
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_offsets);
    cudaFree(d_temp_storage);

    return 0;
}
