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
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I./include
LDFLAGS = 

SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include

# Source files
SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/matrix.cpp \
          $(SRC_DIR)/utils.cpp \
          $(SRC_DIR)/cg.cpp \
          $(SRC_DIR)/bicgstab.cpp \
          $(SRC_DIR)/gmres.cpp \
          $(SRC_DIR)/ca_gmres.cpp \
          $(SRC_DIR)/ca_kernels.cpp \
          $(SRC_DIR)/jacobi.cpp \
          $(SRC_DIR)/block_jacobi.cpp \
          $(SRC_DIR)/ilu0.cpp \
          $(SRC_DIR)/iluk.cpp \
          $(SRC_DIR)/spai.cpp \
          $(SRC_DIR)/additive_schwarz.cpp

OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))

TARGET = solver

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)