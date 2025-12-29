CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CXXFLAGS=-O2 -std=c++17
CPPFLAGS=
CUDAFLAGS=-O2 -std=c++17 -arch=sm_80 -lineinfo -res-usage -src-in-ptx

all: problem3-v0 problem3-v1-cuda problem3-v2-cuda problem3-v3-cuda problem3-v4-thrust

problem3-v0: problem3-v0.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $(patsubst %.c,%.out,$<)

problem3-v1-cuda: problem3-v1-cuda.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-v2-cuda: problem3-v2-cuda.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-v3-cuda: problem3-v3-cuda.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-v4-thrust: problem3-v4-thrust.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

clean:
	rm *.out compile_commands.json
