/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 * 
 * Description:
 *   Parallel implementations of Krylov subspace solvers for sparse linear systems 
 *   using MPI in C++. Solvers included are Conjugate Gradient (CG), BiCGStab, 
 *   and GMRES, with optional preconditioners such as Jacobi, Block Jacobi, and ILU0.
 * 
 *   The project is designed for distributed-memory environments, supporting
 *   CSR-formatted sparse matrices and modular preconditioner interfaces.
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <functional>
#include "matrix.hpp"
#include "preconditioner.hpp"
#include "jacobi.hpp"
#include "cg.hpp"
#include "bicgstab.hpp"
#include "gmres.hpp"

// 定义 solver 描述结构
struct SolverDesc {
    const char* name;
    std::function<int(const CSRMatrix&, const std::vector<double>&, std::vector<double>&,
                      int, double, MPI_Comm, Preconditioner*, int*, double*)> solver;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 构造示例 1D Poisson 矩阵
    int N = 16; // One can define via command line
    CSRMatrix A;
    A.nrows = A.ncols = N;
    A.row_ptr.resize(N+1);
    A.col_idx.resize(3*N-2);
    A.values.resize(3*N-2);

    int idx = 0;
    for (int i = 0; i < N; i++) {
        A.row_ptr[i] = idx;
        if (i > 0) { A.col_idx[idx] = i-1; A.values[idx++] = -1.0; }
        A.col_idx[idx] = i; A.values[idx++] = 2.0;
        if (i < N-1) { A.col_idx[idx] = i+1; A.values[idx++] = -1.0; }
    }
    A.row_ptr[N] = idx;

    std::vector<double> b(N, 1.0);
    std::vector<double> x(N, 0.0);

    JacobiPrecond jacobi(A); // 构造预处理器示例

    // Lambda 包装 solver
    auto cg_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                         std::vector<double>& x, int maxit, double tol,
                         MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return cg_solve(A,b,x,maxit,tol,comm,P,it,res);
    };
    auto bicg_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                           std::vector<double>& x, int maxit, double tol,
                           MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return bicgstab_solve(A,b,x,maxit,tol,comm,P,it,res);
    };
    auto gmres_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                            std::vector<double>& x, int maxit, double tol,
                            MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return gmres_solve(A,b,x,30,maxit,tol,comm,P,it,res);
    };

    std::vector<SolverDesc> solvers = {
        {"CG", cg_wrapper},
        {"BiCGStab", bicg_wrapper},
        {"GMRES", gmres_wrapper}
    };

    
    for (auto& s : solvers) { // 迭代每个 solver
        std::vector<double> x_local(N,0.0);
        int iters = 0;
        double final_norm = 0.0;
        double t0 = MPI_Wtime();
        int status = s.solver(A, b, x_local, 1000, 1e-8, MPI_COMM_WORLD, &jacobi, &iters, &final_norm);
        
        if (status != 0) {
            std::cout << "Solver failed with code " << status << "\n";
        }
        
        double t1 = MPI_Wtime();

        if (rank == 0) {
            std::cout << "Solver=" << s.name
                      << " iters=" << iters
                      << " final_res=" << final_norm
                      << " time=" << (t1 - t0) << " s\n";
        }

    }

    MPI_Finalize();
    return 0;
}
