NVCC=nvcc
CUDAFLAGS=-O2 -std=c++17 -arch=sm_80

all: problem2

problem2: problem2.cu
	$(NVCC) $(CUDAFLAGS) $< -o problem2

clean:
	rm -f problem2

run: problem2
	./problem2

.PHONY: all clean run
