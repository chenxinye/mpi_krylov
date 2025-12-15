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

CXX      := mpicxx
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Iinclude -MMD -MP -fopenmp

LDFLAGS  := -fopenmp
LDLIBS   := 

SRC_DIR := src
OBJ_DIR := obj
BIN     := solver

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))
DEPS := $(OBJS:.o=.d)

all: $(BIN)

$(BIN): $(OBJS)
	@echo "Linking $@"
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

-include $(DEPS)

# 清理
clean:
	@echo "Cleaning..."
	rm -rf $(OBJ_DIR) $(BIN)

.PHONY: all clean
