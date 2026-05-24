# =============================================================================
# MPI Krylov Solver Project - Makefile
#
# Author: Xinye Chen
# Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
#
# Description:
#   Build system for the MPI Krylov Solver Project. Compiles C++17 source files
#   implementing parallel Krylov subspace solvers (CG, BiCGStab, GMRES, CA-GMRES) and 
#   optional preconditioners (Jacobi, Block Jacobi, ILU0) using MPI. 
#
# Usage:
#   make        # Compile all source files and build the executable 'solver'
#   make clean  # Remove object files and executable
# ============================================================================
CXX = mpic++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I./include -march=native
LDFLAGS = 

SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include

# Complete source file list
SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/matrix.cpp \
          $(SRC_DIR)/utils.cpp \
          $(SRC_DIR)/cg.cpp \
          $(SRC_DIR)/bicgstab.cpp \
          $(SRC_DIR)/gmres.cpp \
          $(SRC_DIR)/ca_cg.cpp \
          $(SRC_DIR)/ca_bicgstab.cpp \
          $(SRC_DIR)/ca_gmres.cpp \
          $(SRC_DIR)/ca_kernels.cpp \
          $(SRC_DIR)/pipelined_gmres.cpp \
          $(SRC_DIR)/jacobi.cpp \
          $(SRC_DIR)/block_jacobi.cpp \
          $(SRC_DIR)/ilu0.cpp \
          $(SRC_DIR)/iluk.cpp \
          $(SRC_DIR)/spai.cpp \
          $(SRC_DIR)/additive_schwarz.cpp \
          $(SRC_DIR)/polynomial_precond.cpp

OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))

TARGET = solver

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

test: $(TARGET)
	@echo "Running MPI Krylov Solver Tests..."
	@echo "Test 1: 2 processes"
	mpirun -np 2 ./$(TARGET)
	@echo "\nTest 2: 4 processes"
	mpirun -np 4 ./$(TARGET)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

help:
	@echo "Available targets:"
	@echo "  all   - Build the solver (default)"
	@echo "  test  - Run tests with 2 and 4 processes"
	@echo "  clean - Remove build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make"
	@echo "  mpirun -np 4 ./solver"