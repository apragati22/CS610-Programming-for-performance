NVCC=nvcc
CUDAFLAGS=-O2 -std=c++17 -arch=sm_80

TARGETS=part1_naive part2_shmem part3_optimized part4_pinned

all: $(TARGETS)

part1_naive: part1_naive.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

part2_shmem: part2_shmem.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

part3_optimized: part3_optimized.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

part4_pinned: part4_pinned.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

run: all
	@echo "=== Part 1: Naive Kernel ==="
	./part1_naive
	@echo "\n=== Part 2: Shared Memory Kernel ==="
	./part2_shmem
	@echo "\n=== Part 3: Optimized Kernel ==="
	./part3_optimized
	@echo "\n=== Part 4: Pinned Memory Kernel ==="
	./part4_pinned

.PHONY: all clean run
