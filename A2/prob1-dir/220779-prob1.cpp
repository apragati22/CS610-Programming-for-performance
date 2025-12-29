#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <papi.h>
#include <unistd.h>

using std::cerr;
using std::cout;
using std::endl;
using std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t;

#define INP_H (1 << 6)
#define INP_W (1 << 6)
#define INP_D (1 << 6)
#define FIL_H (3)
#define FIL_W (3)
#define FIL_D (3)

// Cache flushing function to ensure fair measurements
void flush_cache() {
    const size_t cache_size = 64 * 1024 * 1024; // 64MB - larger than typical L3 cache
    volatile char* dummy = new char[cache_size];
    
    // Write to memory to flush cache
    for (size_t i = 0; i < cache_size; i += 64) { // 64-byte cache line size
        dummy[i] = rand() & 0xFF;
    }
    
    // Read from memory to ensure cache is polluted
    volatile char sum = 0;
    for (size_t i = 0; i < cache_size; i += 64) {
        sum += dummy[i];
    }

    
    delete[] dummy;
    
    sync();
    asm volatile("mfence" ::: "memory"); // Memory fence to ensure all operations complete
}

/** Cross-correlation without padding */
void cc_3d_no_padding(const uint64_t* input,
                      const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t* result,
                      const uint64_t outputHeight, const uint64_t outputWidth,
                      const uint64_t outputDepth) {
  for (uint64_t i = 0; i < outputHeight; i++) {
    for (uint64_t j = 0; j < outputWidth; j++) {
      for (uint64_t k = 0; k < outputDepth; k++) {
        uint64_t sum = 0;
        for (uint64_t ki = 0; ki < FIL_H; ki++) {
          for (uint64_t kj = 0; kj < FIL_W; kj++) {
            for (uint64_t kk = 0; kk < FIL_D; kk++) {
              sum += input[(i + ki) * INP_W * INP_D + (j + kj) * INP_D +
                           (k + kk)] *
                     kernel[ki][kj][kk];
            }
          }
        }
        result[i * outputWidth * outputDepth + j * outputDepth + k] += sum;
      }
    }
  }
}

/** Cross-correlation without padding, with blocking */
void cc_3d_no_padding_blocked(const uint64_t* input,
                              const uint64_t (*kernel)[FIL_W][FIL_D],
                              uint64_t* result, const uint64_t outputHeight,
                              const uint64_t outputWidth,
                              const uint64_t outputDepth, const uint64_t i,
                              const uint64_t j, const uint64_t k,
                              const uint64_t curBlockH,
                              const uint64_t curBlockW,
                              const uint64_t curBlockD) {
  for (uint64_t bi = 0; bi < curBlockH; bi++) {
    for (uint64_t bj = 0; bj < curBlockW; bj++) {
      for (uint64_t bk = 0; bk < curBlockD; bk++) {
        uint64_t sum = 0;
        for (uint64_t ki = 0; ki < FIL_H; ki++) {
          for (uint64_t kj = 0; kj < FIL_W; kj++) {
            for (uint64_t kk = 0; kk < FIL_D; kk++) {
              sum += input[(i + bi + ki) * INP_W * INP_D +
                           (j + bj + kj) * INP_D + (k + bk + kk)] *
                     kernel[ki][kj][kk];
            }
          }
        }
        result[(i + bi) * outputWidth * outputDepth +
               (j + bj) * outputDepth + (k + bk)] += sum;
      }
    }
  }
}

// Helper to check PAPI errors
void papi_check(int retval, const char* msg) {
    if (retval != PAPI_OK) {
        std::cerr << "PAPI error " << retval << " at " << msg << ": " << PAPI_strerror(retval) << std::endl;
        exit(1);
    }
}

int main() {
  uint64_t* input = new uint64_t[INP_H * INP_W * INP_D];
  std::fill_n(input, INP_H * INP_W * INP_D, 1);

  uint64_t filter[FIL_H][FIL_W][FIL_D] = {{{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}}};

  uint64_t outputHeight = INP_H - FIL_H + 1;
  uint64_t outputWidth = INP_W - FIL_W + 1;
  uint64_t outputDepth = INP_D - FIL_D + 1;

  double total_naive_time = 0.0;
  double total_blocked_time = 0.0;
  uint64_t bestH = 0, bestW = 0, bestD = 0;
  
  for(int counter = 0; counter<5; counter++){
    
    auto* result = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
    // Flush cache before naive measurement
    flush_cache();

    auto start_naive = std::chrono::high_resolution_clock::now();
    cc_3d_no_padding(input, filter, result, outputHeight, outputWidth, outputDepth);
    auto end_naive = std::chrono::high_resolution_clock::now();
    double naive_time = std::chrono::duration<double>(end_naive - start_naive).count();
    total_naive_time += naive_time;

    double best_time = 1e30;
    // Autotuning for blocked kernel
    for (uint64_t blockHeight : {4, 8, 16, 32}) {
      for (uint64_t blockWidth : {4, 8, 16, 32}) {     
        for (uint64_t blockDepth : {4, 8, 16, 32}) {
          
          auto* result_blocked = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
          
          // Flush cache before each blocked measurement
          flush_cache();
          
          auto start = std::chrono::high_resolution_clock::now();
          for (uint64_t i = 0; i < outputHeight; i += blockHeight) {
            for (uint64_t j = 0; j < outputWidth; j += blockWidth) {
              for (uint64_t k = 0; k < outputDepth; k += blockDepth) {
                uint64_t curBlockH = std::min(blockHeight, outputHeight - i);
                uint64_t curBlockW = std::min(blockWidth, outputWidth - j);
                uint64_t curBlockD = std::min(blockDepth, outputDepth - k);
                cc_3d_no_padding_blocked(input, filter, result_blocked,
                                        outputHeight, outputWidth, outputDepth,
                                        i, j, k,
                                        curBlockH, curBlockW, curBlockD);
              }
            }
          }
          auto end = std::chrono::high_resolution_clock::now();
          double elapsed = std::chrono::duration<double>(end - start).count();

          // Check correctness
          bool correct = true;
          for (uint64_t idx = 0; idx < outputHeight * outputWidth * outputDepth; idx++) {
            if (result[idx] != result_blocked[idx]) {
              correct = false;
              break;
            }
          }
          if (correct && (elapsed < best_time)) {
            best_time = elapsed;
            bestH = blockHeight;
            bestW = blockWidth;
            bestD = blockDepth;
          }
          // cout << "Block size (" << blockHeight << "," << blockWidth << "," << blockDepth
          //     << ") time: " << elapsed << " s" << (correct ? "" : " [INCORRECT]") << endl;
          delete[] result_blocked;
        }
      }
    }
    total_blocked_time += best_time;
    delete[] result;
  }
  cout << "Average naive kernel time: " << total_naive_time/5.0 << " s\n";
  cout << "Average blocked kernel time: " << total_blocked_time/5.0 << " s\n";
  cout << "Speedup: " << (total_naive_time/5.0)/(total_blocked_time/5.0) << "x\n";
  cout << "Best block size : (" << bestH << "," << bestW << "," << bestD << ")\n";
  sleep(5);
  // Allow system to settle before PAPI measurements
  cout << "Starting PAPI measurements...\n";

  // --- PAPI performance measurement ---

  // PAPI setup
  int events[2] = {PAPI_L1_TCM, PAPI_L2_DCM}; // L1 and L2 data cache misses
  int events_blocked[2] = {PAPI_L1_TCM, PAPI_L2_DCM}; // L1 and L2 data cache misses
  long long values[2];

  // papi_check(PAPI_library_init(PAPI_VER_CURRENT), "PAPI_library_init");
  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        std::cerr << "PAPI init error\n";
        return 1;
    }

  // --- Measure naive kernel with PAPI ---
  auto* result_naive = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
  int EventSet = PAPI_NULL;
  PAPI_create_eventset(&EventSet);
  PAPI_add_events(EventSet, events, 2);

  // Flush cache before PAPI naive measurement
  flush_cache();
  
  PAPI_start(EventSet);
  cc_3d_no_padding(input, filter, result_naive, outputHeight, outputWidth, outputDepth);
  PAPI_stop(EventSet, values);
  
  cout << "Naive L1 D-cache misses: " << values[0] << endl;
  cout << "Naive L2 D-cache misses: " << values[1] << endl;

  sleep(5);
  // Flush cache before PAPI blocked measurement
  flush_cache();

  // --- Measure blocked kernel with PAPI ---
  auto* result_blocked_papi = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
  long long values_blocked[2];
  int EventSet_blocked = PAPI_NULL;
  PAPI_create_eventset(&EventSet_blocked);
  PAPI_add_events(EventSet_blocked, events_blocked, 2);

  PAPI_start(EventSet_blocked);
  for (uint64_t i = 0; i < outputHeight; i += bestH) {
      for (uint64_t j = 0; j < outputWidth; j += bestW) {
          for (uint64_t k = 0; k < outputDepth; k += bestD) {
              uint64_t curBlockH = std::min(bestH, outputHeight - i);
              uint64_t curBlockW = std::min(bestW, outputWidth - j);
              uint64_t curBlockD = std::min(bestD, outputDepth - k);
              cc_3d_no_padding_blocked(input, filter, result_blocked_papi,
                                       outputHeight, outputWidth, outputDepth,
                                       i, j, k,
                                       curBlockH, curBlockW, curBlockD);
          }
      }
  }
  PAPI_stop(EventSet_blocked, values_blocked);

  cout << "Blocked L1 D-cache misses: " << values_blocked[0] << endl;
  cout << "Blocked L2 D-cache misses: " << values_blocked[1] << endl;

  delete[] result_blocked_papi;
  delete[] result_naive;
  delete[] input;
  PAPI_shutdown();
  
  return EXIT_SUCCESS;
}
