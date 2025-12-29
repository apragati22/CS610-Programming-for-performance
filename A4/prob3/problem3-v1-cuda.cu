#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)
#define CHUNK_SIZE (1ULL << 28)  // Process 256M iterations per chunk

// CUDA error checking macro
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

// Structure to hold grid parameters
struct GridParams {
  double starts[10];   
  double steps[10];   
  int sizes[10];       
};

// Structure to hold constraint parameters
struct ConstraintParams {
  double coeffs[10][10];  
  double d[10];           
  double epsilon[10];     
};

// Structure to hold result values
struct ResultPoint {
  double x[10];
};

// CUDA kernel for grid search - processes iteration space linearly
__global__ void grid_search_kernel(
    GridParams grid,
    ConstraintParams constraints,
    unsigned long long* result_indices, 
    unsigned long long* result_count,
    unsigned long long chunk_start,
    unsigned long long chunk_end) {
  
  unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long total_threads = gridDim.x * blockDim.x;
  
  unsigned long long iters_per_thread = (chunk_end - chunk_start + total_threads - 1) / total_threads;
  unsigned long long iter_start = chunk_start + iters_per_thread * tid;
  unsigned long long iter_end = min(chunk_end, iter_start + iters_per_thread);
  
  if (iter_start >= chunk_end) return;
  
  for (unsigned long long iter = iter_start; iter < iter_end; ++iter) {
    unsigned long long temp = iter;
    int idx[10];
    for (int i = 9; i >= 0; --i) {
      idx[i] = temp % grid.sizes[i];
      temp /= grid.sizes[i];
    }
    
    // Compute x values from indices
    double x[10];
    for (int i = 0; i < 10; ++i) {
      x[i] = grid.starts[i] + idx[i] * grid.steps[i];
    }
    
    bool satisfied = true;
    for (int c = 0; c < 10; ++c) {
      double sum = 0.0;
      for (int i = 0; i < 10; ++i) {
        sum += constraints.coeffs[c][i] * x[i];
      }
      double diff = fabs(sum - constraints.d[c]);
      
      if (diff > constraints.epsilon[c]) {
        satisfied = false;
        break;
      }
    }
    
    if (satisfied) {
      unsigned long long pos = atomicAdd(result_count, 1ULL);
      result_indices[pos] = iter;
    }
  }
}

// Kernel to reconstruct results from iteration indices
__global__ void reconstruct_results_kernel(
    unsigned long long* indices,
    GridParams grid,
    ResultPoint* results,
    unsigned long long count) {
  
  unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= count) return;
  
  unsigned long long iter = indices[tid];
  unsigned long long temp = iter;
  
  int idx[10];
  for (int i = 9; i >= 0; --i) {
    idx[i] = temp % grid.sizes[i];
    temp /= grid.sizes[i];
  }
  
  for (int i = 0; i < 10; ++i) {
    results[tid].x[i] = grid.starts[i] + idx[i] * grid.steps[i];
  }
}

// Host function to set up and launch kernel
void gridloopsearch_cuda(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6,
    double dd7, double dd8, double dd9, double dd10, double dd11, double dd12,
    double dd13, double dd14, double dd15, double dd16, double dd17,
    double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27,
    double dd28, double dd29, double dd30, double c11, double c12, double c13,
    double c14, double c15, double c16, double c17, double c18, double c19,
    double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29,
    double c210, double d2, double ey2, double c31, double c32, double c33,
    double c34, double c35, double c36, double c37, double c38, double c39,
    double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49,
    double c410, double d4, double ey4, double c51, double c52, double c53,
    double c54, double c55, double c56, double c57, double c58, double c59,
    double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69,
    double c610, double d6, double ey6, double c71, double c72, double c73,
    double c74, double c75, double c76, double c77, double c78, double c79,
    double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89,
    double c810, double d8, double ey8, double c91, double c92, double c93,
    double c94, double c95, double c96, double c97, double c98, double c99,
    double c910, double d9, double ey9, double c101, double c102, double c103,
    double c104, double c105, double c106, double c107, double c108,
    double c109, double c1010, double d10, double ey10, double kk) {
  
  // Prepare grid parameters
  GridParams h_grid;
  h_grid.starts[0] = dd1; h_grid.steps[0] = dd3;
  h_grid.starts[1] = dd4; h_grid.steps[1] = dd6;
  h_grid.starts[2] = dd7; h_grid.steps[2] = dd9;
  h_grid.starts[3] = dd10; h_grid.steps[3] = dd12;
  h_grid.starts[4] = dd13; h_grid.steps[4] = dd15;
  h_grid.starts[5] = dd16; h_grid.steps[5] = dd18;
  h_grid.starts[6] = dd19; h_grid.steps[6] = dd21;
  h_grid.starts[7] = dd22; h_grid.steps[7] = dd24;
  h_grid.starts[8] = dd25; h_grid.steps[8] = dd27;
  h_grid.starts[9] = dd28; h_grid.steps[9] = dd30;
  
  h_grid.sizes[0] = (int)floor((dd2 - dd1) / dd3);
  h_grid.sizes[1] = (int)floor((dd5 - dd4) / dd6);
  h_grid.sizes[2] = (int)floor((dd8 - dd7) / dd9);
  h_grid.sizes[3] = (int)floor((dd11 - dd10) / dd12);
  h_grid.sizes[4] = (int)floor((dd14 - dd13) / dd15);
  h_grid.sizes[5] = (int)floor((dd17 - dd16) / dd18);
  h_grid.sizes[6] = (int)floor((dd20 - dd19) / dd21);
  h_grid.sizes[7] = (int)floor((dd23 - dd22) / dd24);
  h_grid.sizes[8] = (int)floor((dd26 - dd25) / dd27);
  h_grid.sizes[9] = (int)floor((dd29 - dd28) / dd30);
  
  // Calculate total iterations
  unsigned long long total_iters = 1ULL;
  for (int i = 0; i < 10; i++) {
    total_iters *= h_grid.sizes[i];
  }
  
  // Prepare constraint parameters
  ConstraintParams h_cons;
  h_cons.coeffs[0][0] = c11; h_cons.coeffs[0][1] = c12; h_cons.coeffs[0][2] = c13;
  h_cons.coeffs[0][3] = c14; h_cons.coeffs[0][4] = c15; h_cons.coeffs[0][5] = c16;
  h_cons.coeffs[0][6] = c17; h_cons.coeffs[0][7] = c18; h_cons.coeffs[0][8] = c19;
  h_cons.coeffs[0][9] = c110; h_cons.d[0] = d1; h_cons.epsilon[0] = kk * ey1;
  
  h_cons.coeffs[1][0] = c21; h_cons.coeffs[1][1] = c22; h_cons.coeffs[1][2] = c23;
  h_cons.coeffs[1][3] = c24; h_cons.coeffs[1][4] = c25; h_cons.coeffs[1][5] = c26;
  h_cons.coeffs[1][6] = c27; h_cons.coeffs[1][7] = c28; h_cons.coeffs[1][8] = c29;
  h_cons.coeffs[1][9] = c210; h_cons.d[1] = d2; h_cons.epsilon[1] = kk * ey2;
  
  h_cons.coeffs[2][0] = c31; h_cons.coeffs[2][1] = c32; h_cons.coeffs[2][2] = c33;
  h_cons.coeffs[2][3] = c34; h_cons.coeffs[2][4] = c35; h_cons.coeffs[2][5] = c36;
  h_cons.coeffs[2][6] = c37; h_cons.coeffs[2][7] = c38; h_cons.coeffs[2][8] = c39;
  h_cons.coeffs[2][9] = c310; h_cons.d[2] = d3; h_cons.epsilon[2] = kk * ey3;
  
  h_cons.coeffs[3][0] = c41; h_cons.coeffs[3][1] = c42; h_cons.coeffs[3][2] = c43;
  h_cons.coeffs[3][3] = c44; h_cons.coeffs[3][4] = c45; h_cons.coeffs[3][5] = c46;
  h_cons.coeffs[3][6] = c47; h_cons.coeffs[3][7] = c48; h_cons.coeffs[3][8] = c49;
  h_cons.coeffs[3][9] = c410; h_cons.d[3] = d4; h_cons.epsilon[3] = kk * ey4;
  
  h_cons.coeffs[4][0] = c51; h_cons.coeffs[4][1] = c52; h_cons.coeffs[4][2] = c53;
  h_cons.coeffs[4][3] = c54; h_cons.coeffs[4][4] = c55; h_cons.coeffs[4][5] = c56;
  h_cons.coeffs[4][6] = c57; h_cons.coeffs[4][7] = c58; h_cons.coeffs[4][8] = c59;
  h_cons.coeffs[4][9] = c510; h_cons.d[4] = d5; h_cons.epsilon[4] = kk * ey5;
  
  h_cons.coeffs[5][0] = c61; h_cons.coeffs[5][1] = c62; h_cons.coeffs[5][2] = c63;
  h_cons.coeffs[5][3] = c64; h_cons.coeffs[5][4] = c65; h_cons.coeffs[5][5] = c66;
  h_cons.coeffs[5][6] = c67; h_cons.coeffs[5][7] = c68; h_cons.coeffs[5][8] = c69;
  h_cons.coeffs[5][9] = c610; h_cons.d[5] = d6; h_cons.epsilon[5] = kk * ey6;
  
  h_cons.coeffs[6][0] = c71; h_cons.coeffs[6][1] = c72; h_cons.coeffs[6][2] = c73;
  h_cons.coeffs[6][3] = c74; h_cons.coeffs[6][4] = c75; h_cons.coeffs[6][5] = c76;
  h_cons.coeffs[6][6] = c77; h_cons.coeffs[6][7] = c78; h_cons.coeffs[6][8] = c79;
  h_cons.coeffs[6][9] = c710; h_cons.d[6] = d7; h_cons.epsilon[6] = kk * ey7;
  
  h_cons.coeffs[7][0] = c81; h_cons.coeffs[7][1] = c82; h_cons.coeffs[7][2] = c83;
  h_cons.coeffs[7][3] = c84; h_cons.coeffs[7][4] = c85; h_cons.coeffs[7][5] = c86;
  h_cons.coeffs[7][6] = c87; h_cons.coeffs[7][7] = c88; h_cons.coeffs[7][8] = c89;
  h_cons.coeffs[7][9] = c810; h_cons.d[7] = d8; h_cons.epsilon[7] = kk * ey8;
  
  h_cons.coeffs[8][0] = c91; h_cons.coeffs[8][1] = c92; h_cons.coeffs[8][2] = c93;
  h_cons.coeffs[8][3] = c94; h_cons.coeffs[8][4] = c95; h_cons.coeffs[8][5] = c96;
  h_cons.coeffs[8][6] = c97; h_cons.coeffs[8][7] = c98; h_cons.coeffs[8][8] = c99;
  h_cons.coeffs[8][9] = c910; h_cons.d[8] = d9; h_cons.epsilon[8] = kk * ey9;
  
  h_cons.coeffs[9][0] = c101; h_cons.coeffs[9][1] = c102; h_cons.coeffs[9][2] = c103;
  h_cons.coeffs[9][3] = c104; h_cons.coeffs[9][4] = c105; h_cons.coeffs[9][5] = c106;
  h_cons.coeffs[9][6] = c107; h_cons.coeffs[9][7] = c108; h_cons.coeffs[9][8] = c109;
  h_cons.coeffs[9][9] = c1010; h_cons.d[9] = d10; h_cons.epsilon[9] = kk * ey10;
  
  // Allocate device memory for chunked processing
  unsigned long long max_per_chunk = CHUNK_SIZE;
  unsigned long long* d_indices;
  unsigned long long* d_count;
  ResultPoint* d_results;
  
  cudaCheckError(cudaMalloc(&d_indices, max_per_chunk * sizeof(unsigned long long)));
  cudaCheckError(cudaMalloc(&d_count, sizeof(unsigned long long)));
  cudaCheckError(cudaMalloc(&d_results, max_per_chunk * sizeof(ResultPoint)));
  
  // Host buffers
  unsigned long long* h_indices = (unsigned long long*)malloc(max_per_chunk * sizeof(unsigned long long));
  ResultPoint* h_results = (ResultPoint*)malloc(max_per_chunk * sizeof(ResultPoint));
  
  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));
  float total_kernel_time = 0.0f;
  
  FILE* fptr = fopen("./results-v1-cuda.txt", "w");
  if (fptr == NULL) {
    fprintf(stderr, "Error: Cannot create output file\n");
    exit(EXIT_FAILURE);
  }
  
  unsigned long long total_count = 0;
  unsigned long long num_chunks = (total_iters + CHUNK_SIZE - 1) / CHUNK_SIZE;
  
  // Process in chunks
  for (unsigned long long chunk = 0; chunk < num_chunks; chunk++) {
    unsigned long long chunk_start = chunk * CHUNK_SIZE;
    unsigned long long chunk_end = (chunk + 1) * CHUNK_SIZE;
    if (chunk_end > total_iters) chunk_end = total_iters;
    
    // Reset counter
    cudaCheckError(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    
    // Launch kernel
    int threads = 512;
    unsigned long long chunk_iters = chunk_end - chunk_start;
    int blocks = (chunk_iters + threads - 1) / threads;
    
    cudaCheckError(cudaEventRecord(start));
    grid_search_kernel<<<blocks, threads>>>(h_grid, h_cons, d_indices, d_count, chunk_start, chunk_end);
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaEventSynchronize(stop));
    
    float chunk_time;
    cudaCheckError(cudaEventElapsedTime(&chunk_time, start, stop));
    total_kernel_time += chunk_time;
    
    // Get count
    unsigned long long chunk_count;
    cudaCheckError(cudaMemcpy(&chunk_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    total_count += chunk_count;
    
    if (chunk_count > 0) {
      cudaCheckError(cudaMemcpy(h_indices, d_indices, chunk_count * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      
      // Reconstruct results on CPU for simplicity
      for (unsigned long long i = 0; i < chunk_count; i++) {
        unsigned long long iter = h_indices[i];
        unsigned long long temp = iter;
        
        int idx[10];
        for (int j = 9; j >= 0; --j) {
          idx[j] = temp % h_grid.sizes[j];
          temp /= h_grid.sizes[j];
        }
        
        for (int j = 0; j < 10; ++j) {
          h_results[i].x[j] = h_grid.starts[j] + idx[j] * h_grid.steps[j];
        }
      }
      
      for (unsigned long long i = 0; i < chunk_count; i++) {
        fprintf(fptr, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                h_results[i].x[0], h_results[i].x[1], h_results[i].x[2],
                h_results[i].x[3], h_results[i].x[4], h_results[i].x[5],
                h_results[i].x[6], h_results[i].x[7], h_results[i].x[8],
                h_results[i].x[9]);
      }
    }
  }
  
  fclose(fptr);
  
  printf("V1 kernel time: %.3f s\n", total_kernel_time / 1000.0);
  printf("Total result pnts: %llu\n", total_count);
  
  // Cleanup
  free(h_indices);
  free(h_results);
  cudaFree(d_indices);
  cudaFree(d_count);
  cudaFree(d_results);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

struct timespec begin_grid, end_main;

double a[120];  // disp.txt data
double b[30];   // grid.txt data

int main() {
  int i, j;

  // Read disp.txt
  i = 0;
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open disp.txt\n");
    return 1;
  }

  while (!feof(fp)) {
    if (!fscanf(fp, "%lf", &a[i])) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  // Read grid.txt
  j = 0;
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open grid.txt\n");
    return 1;
  }

  while (!feof(fpq)) {
    if (!fscanf(fpq, "%lf", &b[j])) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fpq);

  double kk = 0.3;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  
  gridloopsearch_cuda(
      b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
      b[12], b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21],
      b[22], b[23], b[24], b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2],
      a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
      a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23],
      a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33],
      a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
      a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53],
      a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63],
      a[64], a[65], a[66], a[67], a[68], a[69], a[70], a[71], a[72], a[73],
      a[74], a[75], a[76], a[77], a[78], a[79], a[80], a[81], a[82], a[83],
      a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91], a[92], a[93],
      a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102], a[103],
      a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
      a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk);
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  printf("V1 end-to-end time: %.6f s\n",
         (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
             (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}
