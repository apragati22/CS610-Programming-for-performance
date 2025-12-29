#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <omp.h>
#include <cassert>

using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const uint64_t TIMESTEPS = 100;

const double W_OWN = (1.0 / 7.0);
const double W_NEIGHBORS = (1.0 / 7.0);

const uint64_t NX = 66; // 64 interior points + 2 boundary points
const uint64_t NY = 66;
const uint64_t NZ = 66;
const uint64_t TOTAL_SIZE = NX * NY * NZ;

const static double EPSILON = std::numeric_limits<double>::epsilon();

// Version 0: Original (baseline)
void stencil_3d_7pt_v0(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 1: LICM optimisation
void stencil_3d_7pt_v1(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    const uint64_t i_offset = i * NY * NZ;
    const uint64_t i_plus_offset = (i + 1) * NY * NZ;
    const uint64_t i_minus_offset = (i - 1) * NY * NZ;
    
    for (int j = 1; j < NY - 1; ++j) {
      const uint64_t jz_offset = j * NZ;
      const uint64_t j_plus_offset = (j + 1) * NZ;
      const uint64_t j_minus_offset = (j - 1) * NZ;
      
      for (int k = 1; k < NZ - 1; ++k) {
        const uint64_t idx = i_offset + jz_offset + k;
        
        double neighbors_sum = 0.0;
        neighbors_sum += curr[i_plus_offset + jz_offset + k];
        neighbors_sum += curr[i_minus_offset + jz_offset + k];
        neighbors_sum += curr[i_offset + j_plus_offset + k];
        neighbors_sum += curr[i_offset + j_minus_offset + k];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];

        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 2: Loop unrolling (innermost k loop, factor 2)
void stencil_3d_7pt_v2(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    const uint64_t i_offset = i * NY * NZ;
    const uint64_t i_plus_offset = (i + 1) * NY * NZ;
    const uint64_t i_minus_offset = (i - 1) * NY * NZ;
    
    for (int j = 1; j < NY - 1; ++j) {
      const uint64_t jz_offset = j * NZ;
      const uint64_t j_plus_offset = (j + 1) * NZ;
      const uint64_t j_minus_offset = (j - 1) * NZ;
      
      int k = 1;
      for (; k < NZ - 2; k += 2) {
        // First iteration
        uint64_t idx = i_offset + jz_offset + k;
        double neighbors_sum = 0.0;
        neighbors_sum += curr[i_plus_offset + jz_offset + k];
        neighbors_sum += curr[i_minus_offset + jz_offset + k];
        neighbors_sum += curr[i_offset + j_plus_offset + k];
        neighbors_sum += curr[i_offset + j_minus_offset + k];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];
        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
        
        // Second iteration
        idx = i_offset + jz_offset + k + 1;
        neighbors_sum = 0.0;
        neighbors_sum += curr[i_plus_offset + jz_offset + k + 1];
        neighbors_sum += curr[i_minus_offset + jz_offset + k + 1];
        neighbors_sum += curr[i_offset + j_plus_offset + k + 1];
        neighbors_sum += curr[i_offset + j_minus_offset + k + 1];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];
        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
      }
      
      // remainder
      for (; k < NZ - 1; ++k) {
        const uint64_t idx = i_offset + jz_offset + k;
        double neighbors_sum = 0.0;
        neighbors_sum += curr[i_plus_offset + jz_offset + k];
        neighbors_sum += curr[i_minus_offset + jz_offset + k];
        neighbors_sum += curr[i_offset + j_plus_offset + k];
        neighbors_sum += curr[i_offset + j_minus_offset + k];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];
        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 3: Loop unrolling (factor 4) + LICM
void stencil_3d_7pt_v3(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    const uint64_t i_offset = i * NY * NZ;
    const uint64_t i_plus_offset = (i + 1) * NY * NZ;
    const uint64_t i_minus_offset = (i - 1) * NY * NZ;
    
    for (int j = 1; j < NY - 1; ++j) {
      const uint64_t jz_offset = j * NZ;
      const uint64_t j_plus_offset = (j + 1) * NZ;
      const uint64_t j_minus_offset = (j - 1) * NZ;
      
      int k = 1;
      // Unroll by 4
      for (; k < NZ - 4; k += 4) {
        for (int unroll = 0; unroll < 4; ++unroll) {
          const uint64_t idx = i_offset + jz_offset + k + unroll;
          double neighbors_sum = 0.0;
          neighbors_sum += curr[i_plus_offset + jz_offset + k + unroll];
          neighbors_sum += curr[i_minus_offset + jz_offset + k + unroll];
          neighbors_sum += curr[i_offset + j_plus_offset + k + unroll];
          neighbors_sum += curr[i_offset + j_minus_offset + k + unroll];
          neighbors_sum += curr[idx + 1];
          neighbors_sum += curr[idx - 1];
          next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
        }
      }
      
      // remainder
      for (; k < NZ - 1; ++k) {
        const uint64_t idx = i_offset + jz_offset + k;
        double neighbors_sum = 0.0;
        neighbors_sum += curr[i_plus_offset + jz_offset + k];
        neighbors_sum += curr[i_minus_offset + jz_offset + k];
        neighbors_sum += curr[i_offset + j_plus_offset + k];
        neighbors_sum += curr[i_offset + j_minus_offset + k];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];
        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 4: Loop Permutation (k-i-j order) - innermost stride-1 access
void stencil_3d_7pt_v4(const double* curr, double* next) {
  // Original: i-j-k (stride in k is 1, stride in j is NZ, stride in i is NY*NZ)
  for (int k = 1; k < NZ - 1; ++k) {
    for (int i = 1; i < NX - 1; ++i) {
      const uint64_t i_offset = i * NY * NZ;
      const uint64_t i_plus_offset = (i + 1) * NY * NZ;
      const uint64_t i_minus_offset = (i - 1) * NY * NZ;
      
      for (int j = 1; j < NY - 1; ++j) {
        const uint64_t idx = i_offset + j * NZ + k;
        
        double neighbors_sum = 0.0;
        neighbors_sum += curr[i_plus_offset + j * NZ + k];
        neighbors_sum += curr[i_minus_offset + j * NZ + k];
        neighbors_sum += curr[i_offset + (j + 1) * NZ + k];
        neighbors_sum += curr[i_offset + (j - 1) * NZ + k];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];

        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 5: Loop Permutation (j-k-i order)
void stencil_3d_7pt_v5(const double* curr, double* next) {
  for (int j = 1; j < NY - 1; ++j) {
    const uint64_t jz_offset = j * NZ;
    const uint64_t j_plus_offset = (j + 1) * NZ;
    const uint64_t j_minus_offset = (j - 1) * NZ;
    
    for (int k = 1; k < NZ - 1; ++k) {
      for (int i = 1; i < NX - 1; ++i) {
        const uint64_t idx = i * NY * NZ + jz_offset + k;
        
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + jz_offset + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + jz_offset + k];
        neighbors_sum += curr[i * NY * NZ + j_plus_offset + k];
        neighbors_sum += curr[i * NY * NZ + j_minus_offset + k];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];

        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}


// Version 6: Loop tiling (blocking)
void stencil_3d_7pt_v6(const double* curr, double* next) {
  const int TILE_I = 8;
  const int TILE_J = 8;
  const int TILE_K = 16;
  
  for (int ii = 1; ii < NX - 1; ii += TILE_I) {
    for (int jj = 1; jj < NY - 1; jj += TILE_J) {
      for (int kk = 1; kk < NZ - 1; kk += TILE_K) {
        int i_end = std::min(ii + TILE_I, (int)(NX - 1));
        int j_end = std::min(jj + TILE_J, (int)(NY - 1));
        int k_end = std::min(kk + TILE_K, (int)(NZ - 1));
        
        for (int i = ii; i < i_end; ++i) {
          const uint64_t i_offset = i * NY * NZ;
          const uint64_t i_plus_offset = (i + 1) * NY * NZ;
          const uint64_t i_minus_offset = (i - 1) * NY * NZ;
          
          for (int j = jj; j < j_end; ++j) {
            const uint64_t jz_offset = j * NZ;
            const uint64_t j_plus_offset = (j + 1) * NZ;
            const uint64_t j_minus_offset = (j - 1) * NZ;
            
            for (int k = kk; k < k_end; ++k) {
              const uint64_t idx = i_offset + jz_offset + k;
              
              double neighbors_sum = 0.0;
              neighbors_sum += curr[i_plus_offset + jz_offset + k];
              neighbors_sum += curr[i_minus_offset + jz_offset + k];
              neighbors_sum += curr[i_offset + j_plus_offset + k];
              neighbors_sum += curr[i_offset + j_minus_offset + k];
              neighbors_sum += curr[idx + 1];
              neighbors_sum += curr[idx - 1];

              next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
            }
          }
        }
      }
    }
  }
}

// Version 7: Loop Permutation + Tiling (k-i-j with blocking)
void stencil_3d_7pt_v7(const double* curr, double* next) {
  const int TILE_K = 16;
  const int TILE_I = 8;
  const int TILE_J = 8;
  
  for (int kk = 1; kk < NZ - 1; kk += TILE_K) {
    for (int ii = 1; ii < NX - 1; ii += TILE_I) {
      for (int jj = 1; jj < NY - 1; jj += TILE_J) {
        int k_end = std::min(kk + TILE_K, (int)(NZ - 1));
        int i_end = std::min(ii + TILE_I, (int)(NX - 1));
        int j_end = std::min(jj + TILE_J, (int)(NY - 1));
        
        for (int k = kk; k < k_end; ++k) {
          for (int i = ii; i < i_end; ++i) {
            const uint64_t i_offset = i * NY * NZ;
            const uint64_t i_plus_offset = (i + 1) * NY * NZ;
            const uint64_t i_minus_offset = (i - 1) * NY * NZ;
            
            for (int j = jj; j < j_end; ++j) {
              const uint64_t idx = i_offset + j * NZ + k;
              
              double neighbors_sum = 0.0;
              neighbors_sum += curr[i_plus_offset + j * NZ + k];
              neighbors_sum += curr[i_minus_offset + j * NZ + k];
              neighbors_sum += curr[i_offset + (j + 1) * NZ + k];
              neighbors_sum += curr[i_offset + (j - 1) * NZ + k];
              neighbors_sum += curr[idx + 1];
              neighbors_sum += curr[idx - 1];

              next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
            }
          }
        }
      }
    }
  }
}


// Version 8: OpenMP parallelization on naive 
void stencil_3d_7pt_v8(const double* curr, double* next) {
  #pragma omp parallel for schedule(static)
  for (int i = 1; i < NX - 1; ++i) {
    const uint64_t i_offset = i * NY * NZ;
    const uint64_t i_plus_offset = (i + 1) * NY * NZ;
    const uint64_t i_minus_offset = (i - 1) * NY * NZ;
    
    for (int j = 1; j < NY - 1; ++j) {
      const uint64_t jz_offset = j * NZ;
      const uint64_t j_plus_offset = (j + 1) * NZ;
      const uint64_t j_minus_offset = (j - 1) * NZ;
      
      for (int k = 1; k < NZ - 1; ++k) {
        const uint64_t idx = i_offset + jz_offset + k;
        
        double neighbors_sum = 0.0;
        neighbors_sum += curr[i_plus_offset + jz_offset + k];
        neighbors_sum += curr[i_minus_offset + jz_offset + k];
        neighbors_sum += curr[i_offset + j_plus_offset + k];
        neighbors_sum += curr[i_offset + j_minus_offset + k];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];

        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 9: Loop Permutation with OpenMP (j-k-i with parallel j)
void stencil_3d_7pt_v9(const double* curr, double* next) {
  #pragma omp parallel for schedule(static)
  for (int j = 1; j < NY - 1; ++j) {
    const uint64_t jz_offset = j * NZ;
    const uint64_t j_plus_offset = (j + 1) * NZ;
    const uint64_t j_minus_offset = (j - 1) * NZ;
    
    for (int k = 1; k < NZ - 1; ++k) {
      for (int i = 1; i < NX - 1; ++i) {
        const uint64_t idx = i * NY * NZ + jz_offset + k;
        
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + jz_offset + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + jz_offset + k];
        neighbors_sum += curr[i * NY * NZ + j_plus_offset + k];
        neighbors_sum += curr[i * NY * NZ + j_minus_offset + k];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];

        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}


// Version 10: OpenMP + Unrolling
void stencil_3d_7pt_v10(const double* curr, double* next) {
  #pragma omp parallel for schedule(static)
  for (int i = 1; i < NX - 1; ++i) {
    const uint64_t i_offset = i * NY * NZ;
    const uint64_t i_plus_offset = (i + 1) * NY * NZ;
    const uint64_t i_minus_offset = (i - 1) * NY * NZ;
    
    for (int j = 1; j < NY - 1; ++j) {
      const uint64_t jz_offset = j * NZ;
      const uint64_t j_plus_offset = (j + 1) * NZ;
      const uint64_t j_minus_offset = (j - 1) * NZ;
      
      int k = 1;
      // Unroll by 4
      for (; k < NZ - 4; k += 4) {
        for (int unroll = 0; unroll < 4; ++unroll) {
          const uint64_t idx = i_offset + jz_offset + k + unroll;
          double neighbors_sum = 0.0;
          neighbors_sum += curr[i_plus_offset + jz_offset + k + unroll];
          neighbors_sum += curr[i_minus_offset + jz_offset + k + unroll];
          neighbors_sum += curr[i_offset + j_plus_offset + k + unroll];
          neighbors_sum += curr[i_offset + j_minus_offset + k + unroll];
          neighbors_sum += curr[idx + 1];
          neighbors_sum += curr[idx - 1];
          next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
        }
      }
      
      // remainder
      for (; k < NZ - 1; ++k) {
        const uint64_t idx = i_offset + jz_offset + k;
        double neighbors_sum = 0.0;
        neighbors_sum += curr[i_plus_offset + jz_offset + k];
        neighbors_sum += curr[i_minus_offset + jz_offset + k];
        neighbors_sum += curr[i_offset + j_plus_offset + k];
        neighbors_sum += curr[i_offset + j_minus_offset + k];
        neighbors_sum += curr[idx + 1];
        neighbors_sum += curr[idx - 1];
        next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// Version 11: OpenMP + Tiling (combined)
void stencil_3d_7pt_v11(const double* curr, double* next) {
  const int TILE_I = 8;
  const int TILE_J = 8;
  const int TILE_K = 16;
  
  #pragma omp parallel for schedule(static) collapse(2)
  for (int ii = 1; ii < NX - 1; ii += TILE_I) {
    for (int jj = 1; jj < NY - 1; jj += TILE_J) {
      for (int kk = 1; kk < NZ - 1; kk += TILE_K) {
        int i_end = std::min(ii + TILE_I, (int)(NX - 1));
        int j_end = std::min(jj + TILE_J, (int)(NY - 1));
        int k_end = std::min(kk + TILE_K, (int)(NZ - 1));
        
        for (int i = ii; i < i_end; ++i) {
          const uint64_t i_offset = i * NY * NZ;
          const uint64_t i_plus_offset = (i + 1) * NY * NZ;
          const uint64_t i_minus_offset = (i - 1) * NY * NZ;
          
          for (int j = jj; j < j_end; ++j) {
            const uint64_t jz_offset = j * NZ;
            const uint64_t j_plus_offset = (j + 1) * NZ;
            const uint64_t j_minus_offset = (j - 1) * NZ;
            
            for (int k = kk; k < k_end; ++k) {
              const uint64_t idx = i_offset + jz_offset + k;
              
              double neighbors_sum = 0.0;
              neighbors_sum += curr[i_plus_offset + jz_offset + k];
              neighbors_sum += curr[i_minus_offset + jz_offset + k];
              neighbors_sum += curr[i_offset + j_plus_offset + k];
              neighbors_sum += curr[i_offset + j_minus_offset + k];
              neighbors_sum += curr[idx + 1];
              neighbors_sum += curr[idx - 1];

              next[idx] = W_OWN * curr[idx] + W_NEIGHBORS * neighbors_sum;
            }
          }
        }
      }
    }
  }
}


// Helper function to run a benchmark and verify correctness
void benchmark_version(const char* version_name, 
                       void (*stencil_func)(const double*, double*),
                       double expected_final, double expected_sum) {
  auto* grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  auto* grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  double* current_grid = grid1;
  double* next_grid = grid2;

  auto start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++) {
    stencil_func(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  auto end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();

  double final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  double total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++) {
    total_sum += current_grid[i];
  }

  cout << "====================================\n";
  cout << version_name << "\n";
  cout << "Time: " << duration << " ms" << endl;
  cout << "Final value at center: " << final << "\n";
  cout << "Total sum: " << total_sum << "\n";

  double final_diff = std::abs(final - expected_final);
  double sum_diff = std::abs(total_sum - expected_sum);
  
  // Assert on final value at center and total sum
  // We are dealing with doubles, so use appropriate epsilon for comparison
  const double tolerance = 1e-9;   
  // C++ style assertions with descriptive messages
  // Check final value at center
  if (final_diff > tolerance) {
    cout << "ASSERTION FAILED: Final value mismatch!\n";
    cout << "  Expected: " << expected_final << "\n";
    cout << "  Got:      " << final << "\n";
    cout << "  Diff:     " << final_diff << "\n";
    std::exit(EXIT_FAILURE);
  }
  
  // Check total sum
  if (sum_diff > tolerance) {
    cout << "ASSERTION FAILED: Total sum mismatch!\n";
    cout << "  Expected: " << expected_sum << "\n";
    cout << "  Got:      " << total_sum << "\n";
    cout << "  Diff:     " << sum_diff << "\n";
    std::exit(EXIT_FAILURE);
  }
  
  // C-style assertions (will be compiled out in release builds without -DNDEBUG)
  assert(final_diff <= tolerance && "Final value at center must match baseline");
  assert(sum_diff <= tolerance && "Total sum must match baseline");
  
  cout << "Correctness verified! (diff < " << tolerance << ")\n";

  delete[] grid1;
  delete[] grid2;
}

int main() {
  cout << "3D Stencil (7-point) Optimization Benchmark\n";
  cout << "Grid size: " << NX << " x " << NY << " x " << NZ << "\n";
  cout << "Timesteps: " << TIMESTEPS << "\n";
  cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
  cout << "====================================\n\n";

  // Run baseline first to get expected values
  auto* grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  auto* grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  double* current_grid = grid1;
  double* next_grid = grid2;

  for (int t = 0; t < TIMESTEPS; t++) {
    stencil_3d_7pt_v0(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }

  double expected_final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  double expected_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++) {
    expected_sum += current_grid[i];
  }

  delete[] grid1;
  delete[] grid2;

  // Benchmark all versions
  benchmark_version("Version 0: Baseline", stencil_3d_7pt_v0, expected_final, expected_sum);
  benchmark_version("Version 1: LICM", stencil_3d_7pt_v1, expected_final, expected_sum);
  benchmark_version("Version 2: Loop Unrolling (factor 2)", stencil_3d_7pt_v2, expected_final, expected_sum);
  benchmark_version("Version 3: Loop Unrolling (factor 4)", stencil_3d_7pt_v3, expected_final, expected_sum);
  benchmark_version("Version 4: Loop Permutation (k-i-j)", stencil_3d_7pt_v4, expected_final, expected_sum);
  benchmark_version("Version 5: Loop Permutation (j-k-i)", stencil_3d_7pt_v5, expected_final, expected_sum);
  benchmark_version("Version 6: Loop Tiling", stencil_3d_7pt_v6, expected_final, expected_sum);
  benchmark_version("Version 7: Loop Permutation + Tiling (k-i-j)", stencil_3d_7pt_v7, expected_final, expected_sum);
  benchmark_version("Version 8: OpenMP", stencil_3d_7pt_v8, expected_final, expected_sum);
  benchmark_version("Version 9: Loop Permutation + OpenMP (j-k-i)", stencil_3d_7pt_v9, expected_final, expected_sum);
  benchmark_version("Version 10: OpenMP + Unrolling", stencil_3d_7pt_v10, expected_final, expected_sum);
  benchmark_version("Version 11: OpenMP + Tiling", stencil_3d_7pt_v11, expected_final, expected_sum);
  
  return EXIT_SUCCESS;
}
