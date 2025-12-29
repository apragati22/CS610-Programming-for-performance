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

__global__ void naive_kernel(const double* in, double* out, uint64_t n) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t k = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (i >= 1 && i < (n - 1) && j >= 1 && j < (n - 1) && k >= 1 && k < (n - 1)) {
    out[i * n * n + j * n + k] = 0.8 * (
      in[(i - 1) * n * n + j * n + k] + 
      in[(i + 1) * n * n + j * n + k] +
      in[i * n * n + (j - 1) * n + k] + 
      in[i * n * n + (j + 1) * n + k] +
      in[i * n * n + j * n + (k - 1)] + 
      in[i * n * n + j * n + (k + 1)]
    );
  }
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
  cout << "CPU time: " << duration << " ms\n";

  cudaEvent_t start, end, ete_start, ete_end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventCreate(&ete_start);
  cudaEventCreate(&ete_end);

  cudaEventRecord(ete_start);

  double *d_in, *d_out;
  cudaCheckError(cudaMalloc((void**)&d_in, SIZE_BYTES));
  cudaCheckError(cudaMalloc((void**)&d_out, SIZE_BYTES));
  
  cudaCheckError(cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemset(d_out, 0, SIZE_BYTES));

  dim3 blockSize(8, 8, 8);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                (N + blockSize.y - 1) / blockSize.y,
                (N + blockSize.z - 1) / blockSize.z);
  
  cudaEventRecord(start);
  naive_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);
  cudaCheckError(cudaGetLastError());
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);
  
  cudaCheckError(cudaMemcpy(h_out_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost));
  
  cudaEventRecord(ete_end);
  cudaEventSynchronize(ete_end);
  
  float ete_time = 0.0f;
  cudaEventElapsedTime(&ete_time, ete_start, ete_end);
  
  cout << "Naive kernel time: " << kernel_time << " ms\n";
  cout << "Naive end-to-end time: " << ete_time << " ms\n";
  check_result(h_out_cpu, h_out_gpu, N);
  
  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_in;
  delete[] h_out_cpu;
  delete[] h_out_gpu;
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaEventDestroy(ete_start);
  cudaEventDestroy(ete_end);

  return EXIT_SUCCESS;
}
