CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CXXFLAGS=-O2 -std=c++17
CPPFLAGS=
CUDAFLAGS=-O2 -std=c++17 -arch=sm_80 -lineinfo -res-usage -src-in-ptx

# Default target
all: problem4

problem4: problem4.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@.out

clean:
	rm -f *.out

run: problem4
	./problem4.out

.PHONY: all clean run
