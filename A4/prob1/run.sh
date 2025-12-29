#!/bin/bash

make

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

mkdir -p p1-results

echo "Running part1_naive..."
./part1_naive > p1-results/part1_naive-output.txt 2>&1
nsys profile --trace=cuda -o p1-results/part1_naive ./part1_naive > /dev/null 2>&1
nsys stats p1-results/part1_naive.nsys-rep > p1-results/part1_naive-stats.txt 2>&1
echo "part1_naive complete."

echo "Running part2_shmem..."
./part2_shmem > p1-results/part2_shmem-output.txt 2>&1
nsys profile --trace=cuda -o p1-results/part2_shmem ./part2_shmem > /dev/null 2>&1
nsys stats p1-results/part2_shmem.nsys-rep > p1-results/part2_shmem-stats.txt 2>&1
echo "part2_shmem complete."

echo "Running part3_optimized..."
./part3_optimized > p1-results/part3_optimized-output.txt 2>&1
nsys profile --trace=cuda -o p1-results/part3_optimized ./part3_optimized > /dev/null 2>&1
nsys stats p1-results/part3_optimized.nsys-rep > p1-results/part3_optimized-stats.txt 2>&1
echo "part3_optimized complete."

echo "Running part4_pinned..."
./part4_pinned > p1-results/part4_pinned-output.txt 2>&1
nsys profile --trace=cuda -o p1-results/part4_pinned ./part4_pinned > /dev/null 2>&1
nsys stats p1-results/part4_pinned.nsys-rep > p1-results/part4_pinned-stats.txt 2>&1
echo "part4_pinned complete."

echo ""
echo "All runs complete. Results in p1-results/"
echo "Nsys stats: p1-results/*-stats.txt"
echo "Program outputs: p1-results/*-output.txt"
