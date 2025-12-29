CC=gcc
CXX=g++

CFLAGS=-O3 -std=c17 -mavx -mavx2
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
# Preprocessing flags, not CPP flags
CPPFLAGS=

LIBS=-pthread -lpapi

# First target is the default target
all: problem1 

BIN=./bin
$(shell mkdir -p $(BIN))

.DEFAULT_TARGET: all

# $@ is a variable that contains the target name.
problem1: 220779-prob1.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

# The target is not a filename, always run the rule even if there is a file
.PHONY: clean
clean:
	rm -f -- bin *.out compile_commands.json
