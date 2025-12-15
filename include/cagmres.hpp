/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 
// implementation (naive dense block invert for demonstration)



#ifndef CAGMRES_HPP
#define CAGMRES_HPP

#include <vector>
#include <mpi.h>
#include "matrix.hpp"
#include "preconditioner.hpp"

/**
 * @brief Communication-Avoiding GMRES Solver
 * * @param A 分布式 CSR 矩阵
 * @param b 右端项向量
 * @param x 初始解向量 (兼输出)
 * @param restart 重启动维度 (通常设为 s 的倍数，或直接忽略如果只做一次 CA 步)
 * @param s Basis step size (一次通信生成的基向量个数，例如 5)
 * @param maxit 最大迭代次数
 * @param tol 收敛容差
 * @param comm MPI 通信子
 * @param P 预处理器指针 (当前 CA 实现可能暂未使用，设为 nullptr 即可)
 * @param iters [输出] 实际迭代次数
 * @param final_res [输出] 最终残差
 * @return int 状态码 (0 表示成功)
 */
int cagmres_solve(const CSRMatrix& A, 
                  const std::vector<double>& b, 
                  std::vector<double>& x,
                  int restart, 
                  int s, 
                  int maxit, 
                  double tol,
                  MPI_Comm comm, 
                  Preconditioner* P, 
                  int* iters, 
                  double* final_res);

#endif // CAGMRES_HPP