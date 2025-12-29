#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <iterator>
#include <vector>
#include <cstring>

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define BLOCK_SIZE 256ULL
#define ELEMENTS_PER_BLOCK (BLOCK_SIZE * 2ULL)

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void blelloch_scan_kernel(const uint32_t* input, uint32_t* output, 
                                      uint32_t* block_sums, uint64_t n) {
  __shared__ uint32_t shared[ELEMENTS_PER_BLOCK];
  
  uint64_t tid = threadIdx.x;
  uint64_t global_idx = blockIdx.x * ELEMENTS_PER_BLOCK + tid;
  
  if (global_idx < n) {
    shared[tid] = input[global_idx];
  } else {
    shared[tid] = 0;
  }
  
  if (global_idx + BLOCK_SIZE < n) {
    shared[tid + BLOCK_SIZE] = input[global_idx + BLOCK_SIZE];
  } else {
    shared[tid + BLOCK_SIZE] = 0;
  }
  __syncthreads();
  
  // Up-sweep phase
  uint32_t offset = 1;
  for (uint32_t d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      uint32_t ai = offset * (2 * tid + 1) - 1;
      uint32_t bi = offset * (2 * tid + 2) - 1;
      shared[bi] += shared[ai];
    }
    offset *= 2;
  }
  
  if (tid == 0) {
    if (block_sums != nullptr) {
      block_sums[blockIdx.x] = shared[ELEMENTS_PER_BLOCK - 1];
    }
    shared[ELEMENTS_PER_BLOCK - 1] = 0;
  }
  
  // Down-sweep phase
  for (uint32_t d = 1; d < ELEMENTS_PER_BLOCK; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      uint32_t ai = offset * (2 * tid + 1) - 1;
      uint32_t bi = offset * (2 * tid + 2) - 1;
      uint32_t t = shared[ai];
      shared[ai] = shared[bi];
      shared[bi] += t;
    }
  }
  __syncthreads();
  
  // Convert exclusive to inclusive and write results
  if (global_idx < n) {
    if (global_idx == 0) {
      output[global_idx] = input[global_idx];
    } else {
      output[global_idx] = shared[tid] + input[global_idx];
    }
  }
  
  if (global_idx + BLOCK_SIZE < n) {
    output[global_idx + BLOCK_SIZE] = shared[tid + BLOCK_SIZE] + input[global_idx + BLOCK_SIZE];
  }
}

__global__ void add_block_sums_kernel(uint32_t* data, const uint32_t* block_sums, uint64_t n) {
  uint64_t global_idx = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x;
  
  if (blockIdx.x > 0) {
    uint32_t increment = block_sums[blockIdx.x - 1];
    if (global_idx < n) {
      data[global_idx] += increment;
    }
    if (global_idx + BLOCK_SIZE < n) {
      data[global_idx + BLOCK_SIZE] += increment;
    }
  }
}

__host__ void check_result(const uint32_t* w_ref, const uint32_t* w_opt,
                           const uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    if (w_ref[i] != w_opt[i]) {
      cout << "Differences found between the two arrays.\n";
      assert(false);
    }
  }
  cout << "No differences found between base and test versions\n";
}

__host__ void inclusive_prefix_sum(const uint32_t* input, uint32_t* output, uint64_t size) {
  output[0] = input[0];
  for (uint64_t i = 1; i < size; i++) {
    output[i] = output[i - 1] + input[i];
  }
}

// Recursive helper for scanning block sums
__host__ void scan_block_sums_recursive(uint32_t* d_sums, uint32_t* d_sums_scanned, uint64_t num) {
  if (num <= ELEMENTS_PER_BLOCK) {
    blelloch_scan_kernel<<<1, BLOCK_SIZE>>>(d_sums, d_sums_scanned, nullptr, num);
    cudaCheckError(cudaGetLastError());
  } else {
    uint64_t next_level_blocks = (num + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    uint32_t *d_next_sums, *d_next_scanned;
    cudaCheckError(cudaMalloc(&d_next_sums, next_level_blocks * sizeof(uint32_t)));
    cudaCheckError(cudaMalloc(&d_next_scanned, next_level_blocks * sizeof(uint32_t)));
    
    blelloch_scan_kernel<<<next_level_blocks, BLOCK_SIZE>>>(d_sums, d_sums_scanned, d_next_sums, num);
    cudaCheckError(cudaGetLastError());
    
    scan_block_sums_recursive(d_next_sums, d_next_scanned, next_level_blocks);
    
    add_block_sums_kernel<<<next_level_blocks, BLOCK_SIZE>>>(d_sums_scanned, d_next_scanned, num);
    cudaCheckError(cudaGetLastError());
    
    cudaFree(d_next_sums);
    cudaFree(d_next_scanned);
  }
}

// Copy-then-execute model implementation
__host__ void cte_sum(const uint32_t* h_input, uint32_t* h_output, uint64_t n, float* kernel_time) {
  uint64_t num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
  
  uint32_t *d_input, *d_output, *d_block_sums, *d_block_sums_scanned;
  cudaCheckError(cudaMalloc(&d_input, n * sizeof(uint32_t)));
  cudaCheckError(cudaMalloc(&d_output, n * sizeof(uint32_t)));
  cudaCheckError(cudaMalloc(&d_block_sums, num_blocks * sizeof(uint32_t)));
  cudaCheckError(cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(uint32_t)));
  
  cudaCheckError(cudaMemcpy(d_input, h_input, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  
  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));
  cudaCheckError(cudaEventRecord(start));
  
  blelloch_scan_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, d_block_sums, n);
  cudaCheckError(cudaGetLastError());
  
  if (num_blocks > 1) {
    scan_block_sums_recursive(d_block_sums, d_block_sums_scanned, num_blocks);
    add_block_sums_kernel<<<num_blocks, BLOCK_SIZE>>>(d_output, d_block_sums_scanned, n);
    cudaCheckError(cudaGetLastError());
  }
  
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(kernel_time, start, stop));
  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));
  
  cudaCheckError(cudaMemcpy(h_output, d_output, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_block_sums);
  cudaFree(d_block_sums_scanned);
}

// UVM implementation without hints
__host__ void uvm_sum(const uint32_t* h_input, uint32_t* h_output, uint64_t n, float* kernel_time) {
  uint64_t num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
  
  uint32_t *uvm_input, *uvm_output, *uvm_block_sums, *uvm_block_sums_scanned;
  cudaCheckError(cudaMallocManaged(&uvm_input, n * sizeof(uint32_t)));
  cudaCheckError(cudaMallocManaged(&uvm_output, n * sizeof(uint32_t)));
  cudaCheckError(cudaMallocManaged(&uvm_block_sums, num_blocks * sizeof(uint32_t)));
  cudaCheckError(cudaMallocManaged(&uvm_block_sums_scanned, num_blocks * sizeof(uint32_t)));
  
  memcpy(uvm_input, h_input, n * sizeof(uint32_t));
  
  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));
  cudaCheckError(cudaEventRecord(start));
  
  blelloch_scan_kernel<<<num_blocks, BLOCK_SIZE>>>(uvm_input, uvm_output, uvm_block_sums, n);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());
  
  if (num_blocks > 1) {
    scan_block_sums_recursive(uvm_block_sums, uvm_block_sums_scanned, num_blocks);
    add_block_sums_kernel<<<num_blocks, BLOCK_SIZE>>>(uvm_output, uvm_block_sums_scanned, n);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
  }
  
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(kernel_time, start, stop));
  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));
  
  memcpy(h_output, uvm_output, n * sizeof(uint32_t));
  
  cudaFree(uvm_input);
  cudaFree(uvm_output);
  cudaFree(uvm_block_sums);
  cudaFree(uvm_block_sums_scanned);
}

// UVM with memory hints and prefetch optimization
__host__ void uvm_sum_optimized(const uint32_t* h_input, uint32_t* h_output, uint64_t n, int device, float* kernel_time) {
  uint64_t num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
  
  uint32_t *uvm_input, *uvm_output, *uvm_block_sums, *uvm_block_sums_scanned;
  cudaCheckError(cudaMallocManaged(&uvm_input, n * sizeof(uint32_t)));
  cudaCheckError(cudaMallocManaged(&uvm_output, n * sizeof(uint32_t)));
  cudaCheckError(cudaMallocManaged(&uvm_block_sums, num_blocks * sizeof(uint32_t)));
  cudaCheckError(cudaMallocManaged(&uvm_block_sums_scanned, num_blocks * sizeof(uint32_t)));
  
  memcpy(uvm_input, h_input, n * sizeof(uint32_t));
  
  // Set memory hints for optimization
  cudaCheckError(cudaMemAdvise(uvm_input, n * sizeof(uint32_t), cudaMemAdviseSetAccessedBy, device));
  cudaCheckError(cudaMemAdvise(uvm_output, n * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, device));
  cudaCheckError(cudaMemAdvise(uvm_block_sums, num_blocks * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, device));
  cudaCheckError(cudaMemAdvise(uvm_block_sums_scanned, num_blocks * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, device));
  
  // Prefetch block sums arrays to device
  cudaCheckError(cudaMemPrefetchAsync(uvm_block_sums, num_blocks * sizeof(uint32_t), device, 0));
  cudaCheckError(cudaMemPrefetchAsync(uvm_block_sums_scanned, num_blocks * sizeof(uint32_t), device, 0));
  
  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));
  cudaCheckError(cudaEventRecord(start));
  
  blelloch_scan_kernel<<<num_blocks, BLOCK_SIZE>>>(uvm_input, uvm_output, uvm_block_sums, n);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());
  
  if (num_blocks > 1) {
    scan_block_sums_recursive(uvm_block_sums, uvm_block_sums_scanned, num_blocks);
    add_block_sums_kernel<<<num_blocks, BLOCK_SIZE>>>(uvm_output, uvm_block_sums_scanned, n);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
  }
  
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(kernel_time, start, stop));
  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));
  
  memcpy(h_output, uvm_output, n * sizeof(uint32_t));
  
  cudaFree(uvm_input);
  cudaFree(uvm_output);
  cudaFree(uvm_block_sums);
  cudaFree(uvm_block_sums_scanned);
}

int main(int argc, char** argv) {
  uint64_t N;
  
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <num_elements>\n";
    cout << "Example: " << argv[0] << " 4000000000\n";
    return EXIT_FAILURE;
  }
  
  N = std::strtoull(argv[1], nullptr, 10);
  
  if (N == 0 || N > (1ULL << 35)) {
    cerr << "Invalid size (must be 1 to ~34B elements)\n";
    return EXIT_FAILURE;
  }
  
  int device = 0;
  cudaDeviceProp prop;
  cudaCheckError(cudaGetDeviceProperties(&prop, device));
  
  cout << "Inclusive Prefix Sum\n";
  cout << "Array size: N = " << N << " elements\n";
  cout << "Memory: " << (N * sizeof(uint32_t) / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
  cout << "GPU: " << prop.name << "\n\n";
  
  bool run_cte = (N < 4000000000ULL);
  if (!run_cte) {
    cout << "Skipping CTE (N >= 4B exceeds GPU memory)\n\n";
  }
  
  auto* h_input = new uint32_t[N];
  auto* h_output_cpu = new uint32_t[N];
  auto* h_output_gpu = new uint32_t[N];
  
  std::fill_n(h_input, N, 1);
  
  auto cpu_start = HR::now();
  inclusive_prefix_sum(h_input, h_output_cpu, N);
  auto cpu_end = HR::now();
  auto cpu_time = duration_cast<milliseconds>(cpu_end - cpu_start).count();
  
  cout << "CPU time: " << cpu_time << " ms\n\n";
  
  // Copy-Then-Execute
  long long cte_time = 0;
  float cte_kernel_time = 0.0f;
  if (run_cte) {
    cout << "Copy-Then-Execute Model:\n";
    auto cte_start = HR::now();
    cte_sum(h_input, h_output_gpu, N, &cte_kernel_time);
    auto cte_end = HR::now();
    cte_time = duration_cast<milliseconds>(cte_end - cte_start).count();
    cout << "  Kernel time: " << cte_kernel_time << " ms\n";
    cout << "  End-to-end time: " << cte_time << " ms\n";
    
    if (N < 4000000000ULL) {
      check_result(h_output_cpu, h_output_gpu, N);
    } else {
      cout << "  Verification skipped\n";
    }
    cout << "\n";
  }
  
  // UVM without hints
  long long uvm_time = 0;
  float uvm_kernel_time = 0.0f;
  if (run_cte) {
    cout << "UVM Model (no hints):\n";
    auto uvm_start = HR::now();
    uvm_sum(h_input, h_output_gpu, N, &uvm_kernel_time);
    auto uvm_end = HR::now();
    uvm_time = duration_cast<milliseconds>(uvm_end - uvm_start).count();
    cout << "  Kernel time: " << uvm_kernel_time << " ms\n";
    cout << "  End-to-end time: " << uvm_time << " ms\n";
    
    if (N < 4000000000ULL) {
      check_result(h_output_cpu, h_output_gpu, N);
    } else {
      cout << "  Verification skipped\n";
    }
    cout << "\n";
  }
  
  // UVM with hints
  float uvm_opt_kernel_time = 0.0f;
  cout << "UVM Model (with hints & prefetch):\n";
  auto uvm_opt_start = HR::now();
  uvm_sum_optimized(h_input, h_output_gpu, N, device, &uvm_opt_kernel_time);
  auto uvm_opt_end = HR::now();
  auto uvm_opt_time = duration_cast<milliseconds>(uvm_opt_end - uvm_opt_start).count();
  cout << "  Kernel time: " << uvm_opt_kernel_time << " ms\n";
  cout << "  End-to-end time: " << uvm_opt_time << " ms\n";
  
  if (N < 4000000000ULL) {
    check_result(h_output_cpu, h_output_gpu, N);
  } else {
    cout << "  Verification skipped\n";
  }
  cout << "\n";
  
  delete[] h_input;
  delete[] h_output_cpu;
  delete[] h_output_gpu;
  
  return EXIT_SUCCESS;
}
