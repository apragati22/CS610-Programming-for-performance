#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

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

const uint64_t N = (64);

__host__ void stencil(const double* in, double* out) {
  for (uint64_t i = 1; i < (N - 1); i++) {
    for (uint64_t j = 1; j < (N - 1); j++) {
      for (uint64_t k = 1; k < (N - 1); k++) {
        out[i * N * N + j * N + k] =
            0.8 *
            (in[(i - 1) * N * N + j * N + k] + in[(i + 1) * N * N + j * N + k] +
             in[i * N * N + (j - 1) * N + k] + in[i * N * N + (j + 1) * N + k] +
             in[i * N * N + j * N + (k - 1)] + in[i * N * N + j * N + (k + 1)]);
      }
    }
  }
}

__host__ void check_result(const double* w_ref, const double* w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        double this_diff =
            w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__global__ void shmem_kernel(const double* in, double* out, uint64_t n) {
  extern __shared__ double s_data[];
  
  uint64_t gi = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t gj = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t gk = blockIdx.z * blockDim.z + threadIdx.z;
  
  uint64_t ti = threadIdx.x;
  uint64_t tj = threadIdx.y;
  uint64_t tk = threadIdx.z;
  
  uint64_t tile_dim_i = blockDim.x + 2;
  uint64_t tile_dim_j = blockDim.y + 2;
  uint64_t tile_dim_k = blockDim.z + 2;
  
  #define S_IDX(i, j, k) ((i) * tile_dim_j * tile_dim_k + (j) * tile_dim_k + (k))
  
  if (gi < n && gj < n && gk < n) {
    s_data[S_IDX(ti + 1, tj + 1, tk + 1)] = in[gi * n * n + gj * n + gk];
  }
  
  if (ti == 0 && gi > 0 && gj < n && gk < n) {
    s_data[S_IDX(0, tj + 1, tk + 1)] = in[(gi - 1) * n * n + gj * n + gk];
  }
  if (ti == blockDim.x - 1 && gi < n - 1 && gj < n && gk < n) {
    s_data[S_IDX(ti + 2, tj + 1, tk + 1)] = in[(gi + 1) * n * n + gj * n + gk];
  }
  
  if (tj == 0 && gi < n && gj > 0 && gk < n) {
    s_data[S_IDX(ti + 1, 0, tk + 1)] = in[gi * n * n + (gj - 1) * n + gk];
  }
  if (tj == blockDim.y - 1 && gi < n && gj < n - 1 && gk < n) {
    s_data[S_IDX(ti + 1, tj + 2, tk + 1)] = in[gi * n * n + (gj + 1) * n + gk];
  }
  
  if (tk == 0 && gi < n && gj < n && gk > 0) {
    s_data[S_IDX(ti + 1, tj + 1, 0)] = in[gi * n * n + gj * n + (gk - 1)];
  }
  if (tk == blockDim.z - 1 && gi < n && gj < n && gk < n - 1) {
    s_data[S_IDX(ti + 1, tj + 1, tk + 2)] = in[gi * n * n + gj * n + (gk + 1)];
  }
  
  __syncthreads();
  
  if (gi >= 1 && gi < (n - 1) && gj >= 1 && gj < (n - 1) && gk >= 1 && gk < (n - 1)) {
    out[gi * n * n + gj * n + gk] = 0.8 * (
      s_data[S_IDX(ti, tj + 1, tk + 1)] +
      s_data[S_IDX(ti + 2, tj + 1, tk + 1)] +
      s_data[S_IDX(ti + 1, tj, tk + 1)] +
      s_data[S_IDX(ti + 1, tj + 2, tk + 1)] +
      s_data[S_IDX(ti + 1, tj + 1, tk)] +
      s_data[S_IDX(ti + 1, tj + 1, tk + 2)]
    );
  }
  
  #undef S_IDX
}

int main() {
  uint64_t NUM_ELEMS = (N * N * N);
  uint64_t SIZE_BYTES = NUM_ELEMS * sizeof(double);

  auto* h_in = new double[NUM_ELEMS];
  auto* h_out_cpu = new double[NUM_ELEMS];
  auto* h_out_gpu = new double[NUM_ELEMS];

  srand(42);
  for (uint64_t i = 0; i < NUM_ELEMS; i++) {
    h_in[i] = static_cast<double>(rand());
  }
  std::fill_n(h_out_cpu, NUM_ELEMS, 0.0);
  std::fill_n(h_out_gpu, NUM_ELEMS, 0.0);

  auto cpu_start = HR::now();
  stencil(h_in, h_out_cpu);
  auto cpu_end = HR::now();
  auto duration = duration_cast<milliseconds>(cpu_end - cpu_start).count();
  cout << "CPU time: " << duration << " ms\n\n";

  cudaEvent_t start, end, ete_start, ete_end, bs_ete_start, bs_ete_end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventCreate(&ete_start);
  cudaEventCreate(&ete_end);
  cudaEventCreate(&bs_ete_start);
  cudaEventCreate(&bs_ete_end);

  cudaEventRecord(ete_start);

  double *d_in, *d_out;
  cudaCheckError(cudaMalloc((void**)&d_in, SIZE_BYTES));
  cudaCheckError(cudaMalloc((void**)&d_out, SIZE_BYTES));
  
  cudaCheckError(cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice));

  cout << "Shared Memory Kernel - Block Size Analysis\n";
  cout << "==========================================\n";
  
  int block_sizes[] = {1, 2, 4, 8};
  float best_time = 1e9;
  int best_bs = 0;
  float best_ete_time = 1e9;
  
  for (int bs_idx = 0; bs_idx < 4; bs_idx++) {
    int bs = block_sizes[bs_idx];
    
    cudaEventRecord(bs_ete_start);
    
    std::fill_n(h_out_gpu, NUM_ELEMS, 0.0);
    cudaCheckError(cudaMemset(d_out, 0, SIZE_BYTES));
    
    dim3 blockSize(bs, bs, bs);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y,
                  (N + blockSize.z - 1) / blockSize.z);
    
    size_t shmem_size = (bs + 2) * (bs + 2) * (bs + 2) * sizeof(double);
    
    cout << "\nBlock: " << bs << "x" << bs << "x" << bs 
         << " (" << (bs*bs*bs) << " threads)\n";
    cout << "Grid: " << gridSize.x << "x" << gridSize.y << "x" << gridSize.z << "\n";
    cout << "Shared memory: " << shmem_size << " bytes\n";
    
    cudaEventRecord(start);
    shmem_kernel<<<gridSize, blockSize, shmem_size>>>(d_in, d_out, N);
    cudaCheckError(cudaGetLastError());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float kernel_time = 0.0f;
    cudaEventElapsedTime(&kernel_time, start, end);
    cout << "Kernel time: " << kernel_time << " ms\n";
    
    cudaCheckError(cudaMemcpy(h_out_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost));
    
    cudaEventRecord(bs_ete_end);
    cudaEventSynchronize(bs_ete_end);
    
    float bs_ete_time = 0.0f;
    cudaEventElapsedTime(&bs_ete_time, bs_ete_start, bs_ete_end);
    cout << "End-to-end time: " << bs_ete_time << " ms\n";
    
    check_result(h_out_cpu, h_out_gpu, N);
    
    if (kernel_time < best_time) {
      best_time = kernel_time;
      best_bs = bs;
      best_ete_time = bs_ete_time;
    }
  }

  cudaEventRecord(ete_end);
  cudaEventSynchronize(ete_end);
  
  float total_ete_time = 0.0f;
  cudaEventElapsedTime(&total_ete_time, ete_start, ete_end);

  cout << "\n==========================================\n";
  cout << "Best configuration: " << best_bs << "x" << best_bs << "x" << best_bs << "\n";
  cout << "Best kernel time: " << best_time << " ms\n";
  cout << "Best end-to-end time: " << best_ete_time << " ms\n";
  cout << "Total end-to-end time (all configs): " << total_ete_time << " ms\n";
  
  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_in;
  delete[] h_out_cpu;
  delete[] h_out_gpu;
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaEventDestroy(ete_start);
  cudaEventDestroy(ete_end);
  cudaEventDestroy(bs_ete_start);
  cudaEventDestroy(bs_ete_end);

  return EXIT_SUCCESS;
}
