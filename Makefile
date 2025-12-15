# =============================================================================
# MPI Krylov Solver Project - Makefile
#
# Author: Xinye Chen
# Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
#
# Description:
#   Build system for the MPI Krylov Solver Project. Compiles C++17 source files
#   implementing parallel Krylov subspace solvers (CG, BiCGStab, GMRES) and 
#   optional preconditioners (Jacobi, Block Jacobi, ILU0) using MPI.
#
# Usage:
#   make        # Compile all source files and build the executable 'solver'
#   make clean  # Remove object files and executable
# ============================================================================

# 编译器设置
CXX      := mpicxx
# 编译选项: 
# -Iinclude: 告诉编译器去 include 文件夹找头文件
# -MMD -MP:  自动生成头文件依赖关系 (.d 文件)
# -fopenmp:  如果你在 SpMV 里用了 #pragma omp，需要加上这个
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Iinclude -MMD -MP -fopenmp

# 链接选项
LDFLAGS  := -fopenmp
LDLIBS   := 

# 目录定义
SRC_DIR := src
OBJ_DIR := obj
BIN     := solver

# 查找所有源文件
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# 将 src/xxx.cpp 替换为 obj/xxx.o
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))
# 依赖文件 (.d)
DEPS := $(OBJS:.o=.d)

# 默认目标
all: $(BIN)

# 链接目标
$(BIN): $(OBJS)
	@echo "Linking $@"
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

# 编译目标
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 创建 obj 目录
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# 引入自动生成的依赖关系 (如果存在)
-include $(DEPS)

# 清理
clean:
	@echo "Cleaning..."
	rm -rf $(OBJ_DIR) $(BIN)

.PHONY: all clean
