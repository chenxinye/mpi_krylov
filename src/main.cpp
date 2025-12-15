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
#include "ilu0.hpp"
#include "cg.hpp"
#include "bicgstab.hpp"
#include "gmres.hpp"
#include "cagmres.hpp" // [NEW] Added CA-GMRES header

// Define solver description structure
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

    // Construct distributed 1D Poisson matrix
    int N = 10000; // Global matrix size
    int local_n = N / size; // Base number of rows per rank
    if (rank == size - 1) local_n += N % size; // Handle remainder
    int row_offset = rank * (N / size); // Global index of first local row

    CSRMatrix A;
    A.nrows = local_n;
    A.ncols = N;
    A.row_offset = row_offset;
    A.row_ptr.resize(local_n + 1);
    std::vector<int> col_idx;
    std::vector<double> values;

    int idx = 0;
    for (int i = 0; i < local_n; ++i) {
        int global_i = row_offset + i;
        A.row_ptr[i] = idx;
        if (global_i > 0) { // Left off-diagonal
            col_idx.push_back(global_i - 1);
            values.push_back(-1.0);
            ++idx;
        }
        col_idx.push_back(global_i); // Diagonal
        values.push_back(2.0);
        ++idx;
        if (global_i < N - 1) { // Right off-diagonal
            col_idx.push_back(global_i + 1);
            values.push_back(-1.0);
            ++idx;
        }
    }
    A.row_ptr[local_n] = idx;
    A.col_idx = col_idx;
    A.values = values;

    // Initialize local vectors
    std::vector<double> b(local_n, 1); // Right-hand side
    std::vector<double> x(local_n, 0.0); // Initial guess

    // Construct ILU(0) preconditioner
    ILU0Precond ilu0(A);

    // Lambda wrappers for solvers
    auto cg_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                         std::vector<double>& x, int maxit, double tol,
                         MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return cg_solve(A, b, x, maxit, tol, comm, P, it, res);
    };

    auto bicg_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                           std::vector<double>& x, int maxit, double tol,
                           MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return bicgstab_solve(A, b, x, maxit, tol, comm, P, it, res);
    };

    auto gmres_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                            std::vector<double>& x, int maxit, double tol,
                            MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return gmres_solve(A, b, x, 100, maxit, tol, comm, P, it, res); // Increased restart
    };

    // [NEW] Wrapper for CA-GMRES
    auto cagmres_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                              std::vector<double>& x, int maxit, double tol,
                              MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        // (A, b, x, restart_size, basis_step_s, maxit, tol, comm, P, it, res)
        // Here we use restart=100 and s=5 (basis step size)
        return cagmres_solve(A, b, x, 100, 10, maxit, tol, comm, P, it, res);
    };

    std::vector<SolverDesc> solvers = {
        {"CG", cg_wrapper},
        {"BiCGStab", bicg_wrapper},
        {"GMRES", gmres_wrapper},
        {"CA-GMRES", cagmres_wrapper} // [NEW] Added to the list
    };

    // Iterate over solvers
    for (auto& s : solvers) {
        std::vector<double> x_local(local_n, 0.0); // Reset initial guess
        int iters = 0;
        double final_norm = 0.0;
        
        // Add barrier to ensure fair timing start for each solver
        MPI_Barrier(MPI_COMM_WORLD); 
        double t0 = MPI_Wtime();
        
        int status = s.solver(A, b, x_local, 5000, 1e-8, MPI_COMM_WORLD, &ilu0, &iters, &final_norm);

        if (status != 0 && rank == 0) {
            std::cout << "Solver " << s.name << " failed with code " << status << "\n";
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