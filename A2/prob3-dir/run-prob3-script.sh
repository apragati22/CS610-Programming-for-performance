#!/bin/bash

# Usage: ./run-prob3-script.sh [output_file]

OUTPUT_FILE="${1:-output-prob3.txt}"
THREAD_COUNTS=(1 2 4 8 16 32 64)

echo "Compiling..."
mkdir -p bin
g++ -O3 -std=c++17 -mavx -mavx2 220779-prob3.cpp -o ./bin/problem3.out -pthread

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Running tests, output -> $OUTPUT_FILE"

# Create output file with header
echo "=== Problem 3 Lock Performance Benchmark ===" > "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Run for each thread count
for threads in "${THREAD_COUNTS[@]}"; do
    echo "Testing $threads threads..."
    
    echo "======================================" >> "$OUTPUT_FILE"
    echo "THREAD COUNT: $threads" >> "$OUTPUT_FILE"
    echo "======================================" >> "$OUTPUT_FILE"
    
    ./bin/problem3.out $threads >> "$OUTPUT_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed with $threads threads" >> "$OUTPUT_FILE"
    fi
    
    echo "" >> "$OUTPUT_FILE"
done

echo "Done. Results in $OUTPUT_FILE"
