#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <chrono>

#define THRESHOLD (1.0e-4)  // Practical threshold for floating point comparison

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
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

// Filter parameters
#define FILTER_RADIUS 1  // For 3x3 filter in 2D, 3x3x3 in 3D
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)  // 3
#define FILTER_SIZE_2D (FILTER_SIZE * FILTER_SIZE)  // 9
#define FILTER_SIZE_3D (FILTER_SIZE * FILTER_SIZE * FILTER_SIZE)  // 27

#define TILE_SIZE 16

__constant__ float d_filter_2D[FILTER_SIZE_2D];
__constant__ float d_filter_3D[FILTER_SIZE_3D];

__global__ void kernel2D_basic(const float* input, float* output, 
                               const float* filter, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row < width && col < width) {
    float sum = 0.0f;
    
    // Apply convolution filter
    for (int fr = -FILTER_RADIUS; fr <= FILTER_RADIUS; fr++) {
      for (int fc = -FILTER_RADIUS; fc <= FILTER_RADIUS; fc++) {
        int inputRow = row + fr;
        int inputCol = col + fc;
        
        // Handle boundary conditions (ghost cells are zero)
        if (inputRow >= 0 && inputRow < width && 
            inputCol >= 0 && inputCol < width) {
          float inputValue = input[inputRow * width + inputCol];
          float filterValue = filter[(fr + FILTER_RADIUS) * FILTER_SIZE + 
                                     (fc + FILTER_RADIUS)];
          sum += inputValue * filterValue;
        }
      }
    }
    
    output[row * width + col] = sum;
  }
}

__global__ void kernel2D_opt(const float* input, float* output, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row < width && col < width) {
    float sum = 0.0f;
    
    // Row -1
    int r = row - 1;
    int c = col - 1;
    if (r >= 0 && c >= 0) sum += input[r * width + c] * d_filter_2D[0];
    c = col;
    if (r >= 0) sum += input[r * width + c] * d_filter_2D[1];
    c = col + 1;
    if (r >= 0 && c < width) sum += input[r * width + c] * d_filter_2D[2];
    
    // Row 0
    r = row;
    c = col - 1;
    if (c >= 0) sum += input[r * width + c] * d_filter_2D[3];
    c = col;
    sum += input[r * width + c] * d_filter_2D[4];
    c = col + 1;
    if (c < width) sum += input[r * width + c] * d_filter_2D[5];
    
    // Row +1
    r = row + 1;
    c = col - 1;
    if (r < width && c >= 0) sum += input[r * width + c] * d_filter_2D[6];
    c = col;
    if (r < width) sum += input[r * width + c] * d_filter_2D[7];
    c = col + 1;
    if (r < width && c < width) sum += input[r * width + c] * d_filter_2D[8];
    
    output[row * width + col] = sum;
  }
}

__global__ void kernel3D_basic(const float* input, float* output, 
                               const float* filter, int size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x < size && y < size && z < size) {
    float sum = 0.0f;
    
    // Apply 3D convolution filter
    for (int fz = -FILTER_RADIUS; fz <= FILTER_RADIUS; fz++) {
      for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
        for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
          int inputX = x + fx;
          int inputY = y + fy;
          int inputZ = z + fz;
          
          // Handle boundary conditions (ghost cells are zero)
          if (inputX >= 0 && inputX < size && 
              inputY >= 0 && inputY < size &&
              inputZ >= 0 && inputZ < size) {
            float inputValue = input[inputZ * size * size + inputY * size + inputX];
            float filterValue = filter[(fz + FILTER_RADIUS) * FILTER_SIZE * FILTER_SIZE +
                                       (fy + FILTER_RADIUS) * FILTER_SIZE +
                                       (fx + FILTER_RADIUS)];
            sum += inputValue * filterValue;
          }
        }
      }
    }
    
    output[z * size * size + y * size + x] = sum;
  }
}

__global__ void kernel3D_opt(const float* input, float* output, int size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x < size && y < size && z < size) {
    float sum = 0.0f;
    int filterIdx = 0;
    
    
    #pragma unroll
    for (int fz = -1; fz <= 1; fz++) {
      int inputZ = z + fz;
      bool zValid = (inputZ >= 0 && inputZ < size);
      
      #pragma unroll
      for (int fy = -1; fy <= 1; fy++) {
        int inputY = y + fy;
        bool yValid = (inputY >= 0 && inputY < size);
        
        #pragma unroll
        for (int fx = -1; fx <= 1; fx++) {
          int inputX = x + fx;
          bool xValid = (inputX >= 0 && inputX < size);
          
          if (xValid && yValid && zValid) {
            sum += input[inputZ * size * size + inputY * size + inputX] * d_filter_3D[filterIdx];
          }
          filterIdx++;
        }
      }
    }
    
    output[z * size * size + y * size + x] = sum;
  }
}

__host__ void check_result(const float* w_ref, const float* w_opt, uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    double this_diff = w_ref[i] - w_opt[i];
    if (std::fabs(this_diff) > THRESHOLD) {
      numdiffs++;
      if (std::fabs(this_diff) > maxdiff) {
        maxdiff = std::fabs(this_diff);
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " differences found (threshold " << THRESHOLD
         << ", max diff " << maxdiff << ")\n";
  } else {
    cout << "Verification passed\n";
  }
}

void print2D(const float* A, uint64_t N) {
  for (uint64_t i = 0; i < N; ++i) {
    for (uint64_t j = 0; j < N; ++j) {
      cout << A[i * N + j] << "\t";
    }
    cout << "\n";
  }
}

void print3D(const float* A, uint64_t N) {
  for (uint64_t i = 0; i < N; ++i) {
    for (uint64_t j = 0; j < N; ++j) {
      for (uint64_t k = 0; k < N; ++k) {
        cout << A[i * N * N + j * N + k] << "\t";
      }
      cout << "\n";
    }
    cout << "\n";
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void run_2D_convolution(uint64_t N) {
  cout << "\n2D Convolution (N=" << N << "):\n";
  
  // Allocate host memory for 2D arrays
  uint64_t size_2D = N * N;
  size_t bytes_2D = size_2D * sizeof(float);
  
  float* h_input_2D = (float*)malloc(bytes_2D);
  float* h_output_2D_basic = (float*)malloc(bytes_2D);
  float* h_output_2D_opt = (float*)malloc(bytes_2D);
  float* h_filter_2D = (float*)malloc(FILTER_SIZE_2D * sizeof(float));
  
  // Initialize input with random values
  for (uint64_t i = 0; i < size_2D; i++) {
    h_input_2D[i] = (float)(rand() % 100) / 10.0f;
  }
  
  // Initialize filter for averaging (all weights equal)
  float filterWeight_2D = 1.0f / FILTER_SIZE_2D;
  for (int i = 0; i < FILTER_SIZE_2D; i++) {
    h_filter_2D[i] = filterWeight_2D;
  }
  
  // Allocate device memory for 2D
  float *d_input_2D, *d_output_2D_basic, *d_output_2D_opt, *d_filter_2D_global;
  cudaCheckError(cudaMalloc(&d_input_2D, bytes_2D));
  cudaCheckError(cudaMalloc(&d_output_2D_basic, bytes_2D));
  cudaCheckError(cudaMalloc(&d_output_2D_opt, bytes_2D));
  cudaCheckError(cudaMalloc(&d_filter_2D_global, FILTER_SIZE_2D * sizeof(float)));
  
  // Copy input and filter to device
  cudaCheckError(cudaMemcpy(d_input_2D, h_input_2D, bytes_2D, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_filter_2D_global, h_filter_2D, 
                           FILTER_SIZE_2D * sizeof(float), cudaMemcpyHostToDevice));
  
  // Synchronize to ensure all previous operations completed
  cudaCheckError(cudaDeviceSynchronize());
  
  // Copy filter to constant memory for optimized kernel
  cudaCheckError(cudaMemcpyToSymbol(d_filter_2D, h_filter_2D, 
                                   FILTER_SIZE_2D * sizeof(float), 0, cudaMemcpyHostToDevice));
  
  // Setup execution configuration for 2D
  dim3 block_2D(16, 16);
  dim3 grid_2D((N + block_2D.x - 1) / block_2D.x, (N + block_2D.y - 1) / block_2D.y);
  
  // Create CUDA events for timing 2D kernels
  cudaEvent_t start_2D, stop_2D, ete_start_2D_basic, ete_end_2D_basic;
  cudaCheckError(cudaEventCreate(&start_2D));
  cudaCheckError(cudaEventCreate(&stop_2D));
  cudaCheckError(cudaEventCreate(&ete_start_2D_basic));
  cudaCheckError(cudaEventCreate(&ete_end_2D_basic));
  
  cudaCheckError(cudaEventRecord(ete_start_2D_basic));
  cudaCheckError(cudaEventRecord(start_2D));
  kernel2D_basic<<<grid_2D, block_2D>>>(d_input_2D, d_output_2D_basic, 
                                        d_filter_2D_global, N);
  cudaCheckError(cudaEventRecord(stop_2D));
  cudaCheckError(cudaEventSynchronize(stop_2D));
  
  float kernel_time_2D_basic = 0.0f;
  cudaCheckError(cudaEventElapsedTime(&kernel_time_2D_basic, start_2D, stop_2D));
  
  cudaCheckError(cudaMemcpy(h_output_2D_basic, d_output_2D_basic, bytes_2D, 
                           cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(ete_end_2D_basic));
  cudaCheckError(cudaEventSynchronize(ete_end_2D_basic));
  
  float ete_time_2D_basic = 0.0f;
  cudaCheckError(cudaEventElapsedTime(&ete_time_2D_basic, ete_start_2D_basic, ete_end_2D_basic));
  
  dim3 block_2D_opt(TILE_SIZE, TILE_SIZE);
  dim3 grid_2D_opt((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
  
  cudaEvent_t ete_start_2D_opt, ete_end_2D_opt;
  cudaCheckError(cudaEventCreate(&ete_start_2D_opt));
  cudaCheckError(cudaEventCreate(&ete_end_2D_opt));
  
  cudaCheckError(cudaEventRecord(ete_start_2D_opt));
  cudaCheckError(cudaEventRecord(start_2D));
  kernel2D_opt<<<grid_2D_opt, block_2D_opt>>>(d_input_2D, d_output_2D_opt, N);
  cudaCheckError(cudaEventRecord(stop_2D));
  cudaCheckError(cudaEventSynchronize(stop_2D));
  
  float kernel_time_2D_opt = 0.0f;
  cudaCheckError(cudaEventElapsedTime(&kernel_time_2D_opt, start_2D, stop_2D));
  
  cudaCheckError(cudaMemcpy(h_output_2D_opt, d_output_2D_opt, bytes_2D, 
                           cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(ete_end_2D_opt));
  cudaCheckError(cudaEventSynchronize(ete_end_2D_opt));
  
  float ete_time_2D_opt = 0.0f;
  cudaCheckError(cudaEventElapsedTime(&ete_time_2D_opt, ete_start_2D_opt, ete_end_2D_opt));
  
  check_result(h_output_2D_basic, h_output_2D_opt, size_2D);
  
  cout << "  Basic - Kernel time: " << kernel_time_2D_basic << " ms\n";
  cout << "  Basic - End-to-end time: " << ete_time_2D_basic << " ms\n";
  cout << "  Optimized - Kernel time: " << kernel_time_2D_opt << " ms\n";
  cout << "  Optimized - End-to-end time: " << ete_time_2D_opt << " ms\n";
  cout << "  Kernel speedup: " << kernel_time_2D_basic / kernel_time_2D_opt << "x\n";
  cout << "  End-to-end speedup: " << ete_time_2D_basic / ete_time_2D_opt << "x\n";
  
  // Clean up 2D events
  cudaEventDestroy(start_2D);
  cudaEventDestroy(stop_2D);
  cudaEventDestroy(ete_start_2D_basic);
  cudaEventDestroy(ete_end_2D_basic);
  cudaEventDestroy(ete_start_2D_opt);
  cudaEventDestroy(ete_end_2D_opt);
  
  // Free memory
  free(h_input_2D);
  free(h_output_2D_basic);
  free(h_output_2D_opt);
  free(h_filter_2D);
  
  cudaFree(d_input_2D);
  cudaFree(d_output_2D_basic);
  cudaFree(d_output_2D_opt);
  cudaFree(d_filter_2D_global);
}

void run_3D_convolution(uint64_t N) {
  cout << "\n3D Convolution (N=" << N << "):\n";
  
  // Allocate host memory for 3D arrays
  uint64_t size_3D = N * N * N;
  size_t bytes_3D = size_3D * sizeof(float);
  
  float* h_input_3D = (float*)malloc(bytes_3D);
  float* h_output_3D_basic = (float*)malloc(bytes_3D);
  float* h_output_3D_opt = (float*)malloc(bytes_3D);
  float* h_filter_3D = (float*)malloc(FILTER_SIZE_3D * sizeof(float));
  
  // Initialize input with random values
  for (uint64_t i = 0; i < size_3D; i++) {
    h_input_3D[i] = (float)(rand() % 100) / 10.0f;
  }
  
  // Initialize filter for averaging (all weights equal)
  float filterWeight_3D = 1.0f / FILTER_SIZE_3D;
  for (int i = 0; i < FILTER_SIZE_3D; i++) {
    h_filter_3D[i] = filterWeight_3D;
  }
  
  // Allocate device memory for 3D
  float *d_input_3D, *d_output_3D_basic, *d_output_3D_opt, *d_filter_3D_global;
  cudaCheckError(cudaMalloc(&d_input_3D, bytes_3D));
  cudaCheckError(cudaMalloc(&d_output_3D_basic, bytes_3D));
  cudaCheckError(cudaMalloc(&d_output_3D_opt, bytes_3D));
  cudaCheckError(cudaMalloc(&d_filter_3D_global, FILTER_SIZE_3D * sizeof(float)));
  
  // Copy input and filter to device
  cudaCheckError(cudaMemcpy(d_input_3D, h_input_3D, bytes_3D, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_filter_3D_global, h_filter_3D, 
                           FILTER_SIZE_3D * sizeof(float), cudaMemcpyHostToDevice));
  
  // Copy filter to constant memory for optimized kernel
  cudaCheckError(cudaMemcpyToSymbol(d_filter_3D, h_filter_3D, 
                                   FILTER_SIZE_3D * sizeof(float), 0, cudaMemcpyHostToDevice));
  
  // Setup execution configuration for 3D
  dim3 block_3D(8, 8, 8);
  dim3 grid_3D((N + block_3D.x - 1) / block_3D.x, 
               (N + block_3D.y - 1) / block_3D.y,
               (N + block_3D.z - 1) / block_3D.z);
  
  // Create CUDA events for timing 3D kernels
  cudaEvent_t start_3D, stop_3D, ete_start_3D_basic, ete_end_3D_basic, ete_start_3D_opt, ete_end_3D_opt;
  cudaCheckError(cudaEventCreate(&start_3D));
  cudaCheckError(cudaEventCreate(&stop_3D));
  cudaCheckError(cudaEventCreate(&ete_start_3D_basic));
  cudaCheckError(cudaEventCreate(&ete_end_3D_basic));
  cudaCheckError(cudaEventCreate(&ete_start_3D_opt));
  cudaCheckError(cudaEventCreate(&ete_end_3D_opt));
  
  cudaCheckError(cudaEventRecord(ete_start_3D_basic));
  cudaCheckError(cudaEventRecord(start_3D));
  kernel3D_basic<<<grid_3D, block_3D>>>(d_input_3D, d_output_3D_basic, 
                                        d_filter_3D_global, N);
  cudaCheckError(cudaEventRecord(stop_3D));
  cudaCheckError(cudaEventSynchronize(stop_3D));
  
  float kernel_time_3D_basic = 0.0f;
  cudaCheckError(cudaEventElapsedTime(&kernel_time_3D_basic, start_3D, stop_3D));
  
  cudaCheckError(cudaMemcpy(h_output_3D_basic, d_output_3D_basic, bytes_3D, 
                           cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(ete_end_3D_basic));
  cudaCheckError(cudaEventSynchronize(ete_end_3D_basic));
  
  float ete_time_3D_basic = 0.0f;
  cudaCheckError(cudaEventElapsedTime(&ete_time_3D_basic, ete_start_3D_basic, ete_end_3D_basic));
  
  cudaCheckError(cudaEventRecord(ete_start_3D_opt));
  cudaCheckError(cudaEventRecord(start_3D));
  kernel3D_opt<<<grid_3D, block_3D>>>(d_input_3D, d_output_3D_opt, N);
  cudaCheckError(cudaEventRecord(stop_3D));
  cudaCheckError(cudaEventSynchronize(stop_3D));
  
  float kernel_time_3D_opt = 0.0f;
  cudaCheckError(cudaEventElapsedTime(&kernel_time_3D_opt, start_3D, stop_3D));
  
  cudaCheckError(cudaMemcpy(h_output_3D_opt, d_output_3D_opt, bytes_3D, 
                           cudaMemcpyDeviceToHost));
  cudaCheckError(cudaEventRecord(ete_end_3D_opt));
  cudaCheckError(cudaEventSynchronize(ete_end_3D_opt));
  
  float ete_time_3D_opt = 0.0f;
  cudaCheckError(cudaEventElapsedTime(&ete_time_3D_opt, ete_start_3D_opt, ete_end_3D_opt));
  
  check_result(h_output_3D_basic, h_output_3D_opt, size_3D);
  
  cout << "  Basic - Kernel time: " << kernel_time_3D_basic << " ms\n";
  cout << "  Basic - End-to-end time: " << ete_time_3D_basic << " ms\n";
  cout << "  Optimized - Kernel time: " << kernel_time_3D_opt << " ms\n";
  cout << "  Optimized - End-to-end time: " << ete_time_3D_opt << " ms\n";
  cout << "  Kernel speedup: " << kernel_time_3D_basic / kernel_time_3D_opt << "x\n";
  cout << "  End-to-end speedup: " << ete_time_3D_basic / ete_time_3D_opt << "x\n";
  
  // Clean up 3D events
  cudaEventDestroy(start_3D);
  cudaEventDestroy(stop_3D);
  cudaEventDestroy(ete_start_3D_basic);
  cudaEventDestroy(ete_end_3D_basic);
  cudaEventDestroy(ete_start_3D_opt);
  cudaEventDestroy(ete_end_3D_opt);
  
  // Free memory
  free(h_input_3D);
  free(h_output_3D_basic);
  free(h_output_3D_opt);
  free(h_filter_3D);
  
  cudaFree(d_input_3D);
  cudaFree(d_output_3D_basic);
  cudaFree(d_output_3D_opt);
  cudaFree(d_filter_3D_global);
}

int main() {
  cout << "Filter size: " << FILTER_SIZE << "x" << FILTER_SIZE 
       << " (2D), " << FILTER_SIZE << "x" << FILTER_SIZE << "x" << FILTER_SIZE << " (3D)\n";
  cout << "========================================\n";
  
  uint64_t sizes_2D[] = {64, 128, 256, 512, 1024};
  cout << "\n=== 2D CONVOLUTION TESTS ===\n";
  for (int i = 0; i < 5; i++) {
    run_2D_convolution(sizes_2D[i]);
  }
  
  uint64_t sizes_3D[] = {32, 64, 128, 256, 512};
  cout << "\n=== 3D CONVOLUTION TESTS ===\n";
  for (int i = 0; i < 5; i++) {
    run_3D_convolution(sizes_3D[i]);
  }

  return EXIT_SUCCESS;
}
