#!/bin/bash

# Usage: ./run-prob2-script.sh <binary_path> <input_file> [output_report_file]

BIN_PATH="$1"
INPUT_FILE="$2"
OUTPUT_FILE="${3:-./perf-report_performance.out}"

if [ -z "$BIN_PATH" ] || [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 <binary_path> <input_file> [output_report_file]"
    echo "Example: $0 ./bin/problem2.out ./prob2-test1/input ./perf-report_performance.out"
    exit 1
fi

if [ ! -x "$BIN_PATH" ]; then
    if [ -f "$BIN_PATH" ]; then
        echo "Warning: '$BIN_PATH' is not executable. Attempting to run may fail."
    else
        echo "Error: Binary '$BIN_PATH' not found."
        exit 1
    fi
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

echo "Starting perf c2c recording..."
if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
    echo "perf_event_paranoid setting: $(cat /proc/sys/kernel/perf_event_paranoid)"
fi

[ -f perf.data ] && rm perf.data

# Record with perf c2c
echo "Recording with: perf c2c record -F 22000 -- $BIN_PATH 5 $INPUT_FILE"
perf c2c record -F 22000 -- "$BIN_PATH" 5 "$INPUT_FILE"

if [ ! -f perf.data ]; then
    echo "Error: perf.data was not created"
    exit 1
fi

# Check if perf.data has data
if [ ! -s perf.data ]; then
    echo "Error: perf.data is empty (zero-sized)"
    echo "This might happen if:"
    echo "  1. The program runs too quickly"
    echo "  2. There's insufficient cache coherency traffic to detect"
    echo "  3. perf_event_paranoid is still too restrictive"
    exit 1
fi

echo "perf.data created successfully ($(stat -c%s perf.data) bytes)"

# Generate the report
echo "Generating perf c2c report -> $OUTPUT_FILE ..."
perf c2c report -NN -i perf.data --stdio > "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Performance report generated successfully: $OUTPUT_FILE"
    echo "Report size: $(wc -l < "$OUTPUT_FILE") lines"
else
    echo "Error generating perf report"
    exit 1
fi