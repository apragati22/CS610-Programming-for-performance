#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)
#define CHUNK_SIZE (1ULL << 28)  // 256M iterations per chunk

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

// Compact structures for constant memory efficiency
struct GridParams {
  double starts[10];
  double steps[10];
  int sizes[10];
};

struct ConstraintParams {
  double coeffs[10][10];
  double d_vals[10];
  double epsilons[10];
};

struct ResultPoint {
  double x[10];
};

__constant__ GridParams c_grid;
__constant__ ConstraintParams c_cons;

struct ConstraintChecker {
  __device__
  bool operator()(unsigned long long iter) const {
    unsigned long long temp = iter;
    int idx[10];
    #pragma unroll
    for (int i = 9; i >= 0; --i) {
      idx[i] = temp % c_grid.sizes[i];
      temp /= c_grid.sizes[i];
    }
    
    double x[10];
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
      x[i] = c_grid.starts[i] + idx[i] * c_grid.steps[i];
    }
    
    // Early exit constraint checking
    for (int c = 0; c < 10; ++c) {
      double sum = 0.0;
      #pragma unroll
      for (int i = 0; i < 10; ++i) {
        sum += c_cons.coeffs[c][i] * x[i];
      }
      if (fabs(sum - c_cons.d_vals[c]) > c_cons.epsilons[c]) {
        return false;
      }
    }
    return true;
  }
};

struct IndexToResult {
  __device__
  ResultPoint operator()(unsigned long long iter) const {
    ResultPoint res;
    unsigned long long temp = iter;
    
    int idx[10];
    #pragma unroll
    for (int i = 9; i >= 0; --i) {
      idx[i] = temp % c_grid.sizes[i];
      temp /= c_grid.sizes[i];
    }
    
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
      res.x[i] = c_grid.starts[i] + idx[i] * c_grid.steps[i];
    }
    
    return res;
  }
};

void gridloopsearch_thrust(
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
  
  // Setup grid parameters in compact structure
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
  
  // Setup constraint parameters
  ConstraintParams h_cons;
  
  h_cons.coeffs[0][0] = c11; h_cons.coeffs[0][1] = c12; h_cons.coeffs[0][2] = c13;
  h_cons.coeffs[0][3] = c14; h_cons.coeffs[0][4] = c15; h_cons.coeffs[0][5] = c16;
  h_cons.coeffs[0][6] = c17; h_cons.coeffs[0][7] = c18; h_cons.coeffs[0][8] = c19;
  h_cons.coeffs[0][9] = c110; h_cons.d_vals[0] = d1; h_cons.epsilons[0] = kk * ey1;
  
  h_cons.coeffs[1][0] = c21; h_cons.coeffs[1][1] = c22; h_cons.coeffs[1][2] = c23;
  h_cons.coeffs[1][3] = c24; h_cons.coeffs[1][4] = c25; h_cons.coeffs[1][5] = c26;
  h_cons.coeffs[1][6] = c27; h_cons.coeffs[1][7] = c28; h_cons.coeffs[1][8] = c29;
  h_cons.coeffs[1][9] = c210; h_cons.d_vals[1] = d2; h_cons.epsilons[1] = kk * ey2;
  
  h_cons.coeffs[2][0] = c31; h_cons.coeffs[2][1] = c32; h_cons.coeffs[2][2] = c33;
  h_cons.coeffs[2][3] = c34; h_cons.coeffs[2][4] = c35; h_cons.coeffs[2][5] = c36;
  h_cons.coeffs[2][6] = c37; h_cons.coeffs[2][7] = c38; h_cons.coeffs[2][8] = c39;
  h_cons.coeffs[2][9] = c310; h_cons.d_vals[2] = d3; h_cons.epsilons[2] = kk * ey3;
  
  h_cons.coeffs[3][0] = c41; h_cons.coeffs[3][1] = c42; h_cons.coeffs[3][2] = c43;
  h_cons.coeffs[3][3] = c44; h_cons.coeffs[3][4] = c45; h_cons.coeffs[3][5] = c46;
  h_cons.coeffs[3][6] = c47; h_cons.coeffs[3][7] = c48; h_cons.coeffs[3][8] = c49;
  h_cons.coeffs[3][9] = c410; h_cons.d_vals[3] = d4; h_cons.epsilons[3] = kk * ey4;
  
  h_cons.coeffs[4][0] = c51; h_cons.coeffs[4][1] = c52; h_cons.coeffs[4][2] = c53;
  h_cons.coeffs[4][3] = c54; h_cons.coeffs[4][4] = c55; h_cons.coeffs[4][5] = c56;
  h_cons.coeffs[4][6] = c57; h_cons.coeffs[4][7] = c58; h_cons.coeffs[4][8] = c59;
  h_cons.coeffs[4][9] = c510; h_cons.d_vals[4] = d5; h_cons.epsilons[4] = kk * ey5;
  
  h_cons.coeffs[5][0] = c61; h_cons.coeffs[5][1] = c62; h_cons.coeffs[5][2] = c63;
  h_cons.coeffs[5][3] = c64; h_cons.coeffs[5][4] = c65; h_cons.coeffs[5][5] = c66;
  h_cons.coeffs[5][6] = c67; h_cons.coeffs[5][7] = c68; h_cons.coeffs[5][8] = c69;
  h_cons.coeffs[5][9] = c610; h_cons.d_vals[5] = d6; h_cons.epsilons[5] = kk * ey6;
  
  h_cons.coeffs[6][0] = c71; h_cons.coeffs[6][1] = c72; h_cons.coeffs[6][2] = c73;
  h_cons.coeffs[6][3] = c74; h_cons.coeffs[6][4] = c75; h_cons.coeffs[6][5] = c76;
  h_cons.coeffs[6][6] = c77; h_cons.coeffs[6][7] = c78; h_cons.coeffs[6][8] = c79;
  h_cons.coeffs[6][9] = c710; h_cons.d_vals[6] = d7; h_cons.epsilons[6] = kk * ey7;
  
  h_cons.coeffs[7][0] = c81; h_cons.coeffs[7][1] = c82; h_cons.coeffs[7][2] = c83;
  h_cons.coeffs[7][3] = c84; h_cons.coeffs[7][4] = c85; h_cons.coeffs[7][5] = c86;
  h_cons.coeffs[7][6] = c87; h_cons.coeffs[7][7] = c88; h_cons.coeffs[7][8] = c89;
  h_cons.coeffs[7][9] = c810; h_cons.d_vals[7] = d8; h_cons.epsilons[7] = kk * ey8;
  
  h_cons.coeffs[8][0] = c91; h_cons.coeffs[8][1] = c92; h_cons.coeffs[8][2] = c93;
  h_cons.coeffs[8][3] = c94; h_cons.coeffs[8][4] = c95; h_cons.coeffs[8][5] = c96;
  h_cons.coeffs[8][6] = c97; h_cons.coeffs[8][7] = c98; h_cons.coeffs[8][8] = c99;
  h_cons.coeffs[8][9] = c910; h_cons.d_vals[8] = d9; h_cons.epsilons[8] = kk * ey9;
  
  h_cons.coeffs[9][0] = c101; h_cons.coeffs[9][1] = c102; h_cons.coeffs[9][2] = c103;
  h_cons.coeffs[9][3] = c104; h_cons.coeffs[9][4] = c105; h_cons.coeffs[9][5] = c106;
  h_cons.coeffs[9][6] = c107; h_cons.coeffs[9][7] = c108; h_cons.coeffs[9][8] = c109;
  h_cons.coeffs[9][9] = c1010; h_cons.d_vals[9] = d10; h_cons.epsilons[9] = kk * ey10;
  
  // Calculate total iterations
  unsigned long long total_iterations = 1ULL;
  for (int i = 0; i < 10; i++) {
    total_iterations *= h_grid.sizes[i];
  }
  
  cudaCheckError(cudaMemcpyToSymbol(c_grid, &h_grid, sizeof(GridParams)));
  cudaCheckError(cudaMemcpyToSymbol(c_cons, &h_cons, sizeof(ConstraintParams)));
  
  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));
  
  unsigned long long total_count = 0;
  unsigned long long num_chunks = (total_iterations + CHUNK_SIZE - 1) / CHUNK_SIZE;
  
  cudaCheckError(cudaEventRecord(start));
  
  thrust::host_vector<ResultPoint> all_results;
  
  // Process in chunks
  for (unsigned long long chunk = 0; chunk < num_chunks; chunk++) {
    unsigned long long chunk_start = chunk * CHUNK_SIZE;
    unsigned long long chunk_end = (chunk + 1) * CHUNK_SIZE;
    if (chunk_end > total_iterations) chunk_end = total_iterations;
    
    thrust::counting_iterator<unsigned long long> iter_begin(chunk_start);
    thrust::counting_iterator<unsigned long long> iter_end(chunk_end);
    
    unsigned long long chunk_size = chunk_end - chunk_start;
    thrust::device_vector<unsigned long long> valid_indices(chunk_size);
    auto new_end = thrust::copy_if(iter_begin, iter_end,
                                    valid_indices.begin(),
                                    ConstraintChecker());
    
    unsigned long long chunk_count = new_end - valid_indices.begin();
    valid_indices.resize(chunk_count);
    total_count += chunk_count;
    
    if (chunk_count > 0) {
      thrust::device_vector<ResultPoint> results(chunk_count);
      thrust::transform(valid_indices.begin(), valid_indices.end(),
                        results.begin(),
                        IndexToResult());
      
      thrust::host_vector<ResultPoint> h_results = results;
      all_results.insert(all_results.end(), h_results.begin(), h_results.end());
    }
  }
  
  cudaCheckError(cudaEventRecord(stop));
  cudaCheckError(cudaEventSynchronize(stop));
  
  float milliseconds = 0;
  cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
  
  printf("V4 kernel time: %.3f s\n", milliseconds / 1000.0);
  printf("Total result points: %llu\n", total_count);
  
  FILE* fptr = fopen("./results-v4-thrust.txt", "w");
  if (fptr == NULL) {
    fprintf(stderr, "Error: Cannot create output file\n");
    exit(EXIT_FAILURE);
  }
  
  for (size_t i = 0; i < all_results.size(); i++) {
    fprintf(fptr, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
            all_results[i].x[0], all_results[i].x[1], all_results[i].x[2],
            all_results[i].x[3], all_results[i].x[4], all_results[i].x[5],
            all_results[i].x[6], all_results[i].x[7], all_results[i].x[8],
            all_results[i].x[9]);
  }
  
  fclose(fptr);
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

struct timespec begin_grid, end_main;

// Arrays to store input data
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

  // Initialize kk parameter
  double kk = 0.3;

  // Print CUDA device info
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("Error: No CUDA devices found\n");
    return 1;
  }
  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  
  gridloopsearch_thrust(
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
  printf("\nTotal time (including I/O) = %.6f seconds\n",
         (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
             (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}
