CC=gcc
CXX=g++

CFLAGS=-O3 -std=c17 -mavx -mavx2
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
# Preprocessing flags, not CPP flags
CPPFLAGS=

LIBS=-pthread -lpapi

# First target is the default target
all: prob2_naive prob2_false prob2_true generate_test_files

BIN=./bin
$(shell mkdir -p $(BIN))

.DEFAULT_TARGET: all

# $@ is a variable that contains the target name.
prob2_naive: prob2_naive.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

prob2_false: 220779-prob2-false.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

prob2_true: 220779-prob2-true.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

generate_test_files: generate_test_files.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

# The target is not a filename, always run the rule even if there is a file
.PHONY: clean
clean:
	rm -f -- bin *.out compile_commands.json
