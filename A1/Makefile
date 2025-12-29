CC=gcc
CXX=g++

CFLAGS=-O3 -std=c17 -mavx -mavx2
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
# Preprocessing flags, not CPP flags
CPPFLAGS=

LIBS=-pthread

# First target is the default target
all: 220779

BIN=./bin
$(shell mkdir -p $(BIN))

.DEFAULT_TARGET: 220779

# $@ is a variable that contains the target name.
220779: problem3-dir/220779.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

# The target is not a filename, always run the rule even if there is a file
.PHONY: clean
clean:
	rm -rf -- bin *.out compile_commands.json
