/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 * 
 * Optimized benchmark suite
 */

#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <functional>
#include <string>
#include <cmath>

#include "matrix.hpp"
#include "preconditioner.hpp"
#include "ilu0.hpp"
#include "jacobi.hpp"
#include "additive_schwarz.hpp"
#include "polynomial_precond.hpp"

#include "cg.hpp"
#include "bicgstab.hpp"
#include "gmres.hpp"
#include "ca_cg.hpp"
#include "ca_bicgstab.hpp"
#include "cagmres.hpp"
#include "pipelined_gmres.hpp"

using SolverFunc = std::function<int(const CSRMatrix&, const std::vector<double>&, 
                                     std::vector<double>&, int, double, MPI_Comm, 
                                     Preconditioner*, int*, double*)>;

void print_header(int rank) {
    if (rank == 0) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "  MPI Krylov Solver - Performance Benchmark\n";
        std::cout << std::string(80, '=') << "\n\n";
    }
}

void print_result(int rank, const std::string& solver_name, const std::string& precond_name,
                  int iters, double final_norm, double time, bool failed = false) {
    if (rank == 0) {
        if (failed) std::cout << "[FAIL] ";
        std::cout << std::left << std::setw(20) << solver_name 
                  << std::setw(16) << precond_name
                  << " it=" << std::setw(4) << iters
                  << " res=" << std::scientific << std::setprecision(2) << final_norm
                  << std::resetiosflags(std::ios::scientific)
                  << " t=" << std::fixed << std::setprecision(4) << time << "s\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    print_header(rank);

    // Problem size
    int N = 1000;  // Reduced for faster testing
    int local_n = N / size;
    if (rank == size - 1) local_n += N % size;
    int row_offset = rank * (N / size);

    // Construct distributed 1D Poisson matrix: -u'' = f
    // Discretized as: -u_{i-1} + 2u_i - u_{i+1} = h^2 * f_i
    CSRMatrix A;
    A.nrows = local_n;
    A.ncols = N;
    A.row_offset = row_offset;
    A.row_ptr.resize(local_n + 1);
    std::vector<int> col_idx;
    std::vector<double> values;

    double h = 1.0 / (N + 1);  // Grid spacing
    int idx = 0;
    for (int i = 0; i < local_n; ++i) {
        int global_i = row_offset + i;
        A.row_ptr[i] = idx;
        if (global_i > 0) {
            col_idx.push_back(global_i - 1);
            values.push_back(-1.0);
            ++idx;
        }
        col_idx.push_back(global_i);
        values.push_back(2.0);
        ++idx;
        if (global_i < N - 1) {
            col_idx.push_back(global_i + 1);
            values.push_back(-1.0);
            ++idx;
        }
    }
    A.row_ptr[local_n] = idx;
    A.col_idx = col_idx;
    A.values = values;

    if (rank == 0) {
        std::cout << "Problem: 1D Poisson equation -u'' = f\n";
        std::cout << "Grid: N=" << N << ", h=" << h << "\n";
        std::cout << "MPI ranks=" << size << ", local rows per rank: " << local_n << "\n\n";
    }

    // Initialize halo exchange
    double t_halo_start = MPI_Wtime();
    initialize_halo_exchange(A, MPI_COMM_WORLD);
    double t_halo_end = MPI_Wtime();
    
    if (rank == 0) {
        std::cout << "Halo exchange initialized in " << (t_halo_end - t_halo_start) << "s\n\n";
    }

    // Construct RHS: b = A * x_true, where x_true = sin(pi * x)
    // This ensures the solution is in the range of A
    std::vector<double> x_true(local_n);
    for (int i = 0; i < local_n; ++i) {
        int global_i = row_offset + i;
        double x_coord = (global_i + 1) * h;
        x_true[i] = std::sin(M_PI * x_coord);
    }
    
    std::vector<double> b(local_n);
    distributed_matvec(A, x_true, b, MPI_COMM_WORLD);
    
    double b_norm = global_norm(b, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "True solution: x(i) = sin(pi * i * h)\n";
        std::cout << "RHS norm: ||b|| = " << b_norm << "\n\n";
    }

    // Test parameters
    int max_iter = 500;
    double tol = 1e-6;  // Relaxed tolerance

    // ============================================================
    // Construct preconditioners ONCE
    // ============================================================
    
    if (rank == 0) {
        std::cout << "Constructing preconditioners...\n";
    }
    
    double t_precond_start = MPI_Wtime();
    
    JacobiPrecond jacobi(A);
    ILU0Precond ilu0(A);
    
    double t_precond_mid = MPI_Wtime();
    if (rank == 0) {
        std::cout << "  Jacobi + ILU(0): " << (t_precond_mid - t_precond_start) << "s\n";
    }
    
    AdditiveSchwarzPrecond asm_precond(A, 0, MPI_COMM_WORLD);
    
    double t_asm_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "  Additive Schwarz: " << (t_asm_end - t_precond_mid) << "s\n";
    }
    
    PolynomialPrecond poly_neumann(A, PolynomialPrecond::NEUMANN, 3, MPI_COMM_WORLD);
    
    double t_poly1_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "  Polynomial-Neumann: " << (t_poly1_end - t_asm_end) << "s\n";
    }
    
    PolynomialPrecond poly_cheby(A, PolynomialPrecond::CHEBYSHEV, 3, MPI_COMM_WORLD);
    
    double t_poly2_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "  Polynomial-Chebyshev: " << (t_poly2_end - t_poly1_end) << "s\n";
        std::cout << "\nTotal preconditioner construction: " 
                  << (t_poly2_end - t_precond_start) << "s\n\n";
    }

    // ============================================================
    // Define solver wrappers
    // ============================================================
    
    SolverFunc cg_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                               std::vector<double>& x, int maxit, double tol,
                               MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return cg_solve(A, b, x, maxit, tol, comm, P, it, res);
    };

    SolverFunc bicgstab_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                                     std::vector<double>& x, int maxit, double tol,
                                     MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return bicgstab_solve(A, b, x, maxit, tol, comm, P, it, res);
    };

    SolverFunc gmres_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                                  std::vector<double>& x, int maxit, double tol,
                                  MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return gmres_solve(A, b, x, 30, maxit, tol, comm, P, it, res);
    };

    SolverFunc ca_cg_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                                  std::vector<double>& x, int maxit, double tol,
                                  MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return ca_cg_solve(A, b, x, 5, maxit, tol, comm, P, it, res);
    };

    SolverFunc ca_gmres_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                                     std::vector<double>& x, int maxit, double tol,
                                     MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return cagmres_solve(A, b, x, 30, 5, maxit, tol, comm, P, it, res);
    };

    SolverFunc pipelined_gmres_wrapper = [](const CSRMatrix& A, const std::vector<double>& b,
                                            std::vector<double>& x, int maxit, double tol,
                                            MPI_Comm comm, Preconditioner* P, int* it, double* res) {
        return pipelined_gmres_solve(A, b, x, 30, maxit, tol, comm, P, it, res);
    };

    // ============================================================
    // Benchmark Suite
    // ============================================================
    
    if (rank == 0) {
        std::cout << std::string(80, '=') << "\n";
        std::cout << "Starting Benchmark\n";
        std::cout << std::string(80, '=') << "\n\n";
        std::cout << std::left << std::setw(20) << "Solver" 
                  << std::setw(16) << "Preconditioner" 
                  << "Performance\n";
        std::cout << std::string(80, '-') << "\n";
    }

    // Test 1: Classical CG
    if (rank == 0) std::cout << "\n--- Conjugate Gradient ---\n";
    
    std::vector<std::pair<std::string, Preconditioner*>> cg_preconds = {
        {"None", nullptr},
        {"Jacobi", &jacobi},
        {"ILU(0)", &ilu0},
        {"Add-Schwarz", &asm_precond}
    };
    
    for (const auto& [name, precond] : cg_preconds) {
        std::vector<double> x(local_n, 0.0);
        int iters = 0;
        double final_norm = 0.0;
        
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        int status = cg_wrapper(A, b, x, max_iter, tol, MPI_COMM_WORLD, 
                               precond, &iters, &final_norm);
        double t1 = MPI_Wtime();
        
        print_result(rank, "CG", name, iters, final_norm, t1 - t0, status != 0);
    }

    // Test 2: CA-CG
    if (rank == 0) std::cout << "\n--- CA-CG (s=5) ---\n";
    
    std::vector<std::pair<std::string, Preconditioner*>> ca_cg_preconds = {
        {"None", nullptr},
        {"Jacobi", &jacobi},
        {"Poly-Neumann", &poly_neumann},
        {"ILU(0)", &ilu0}
    };
    
    for (const auto& [name, precond] : ca_cg_preconds) {
        std::vector<double> x(local_n, 0.0);
        int iters = 0;
        double final_norm = 0.0;
        
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        int status = ca_cg_wrapper(A, b, x, max_iter, tol, MPI_COMM_WORLD, 
                                   precond, &iters, &final_norm);
        double t1 = MPI_Wtime();
        
        print_result(rank, "CA-CG", name, iters, final_norm, t1 - t0, status != 0);
    }

    // Test 3: GMRES variants
    if (rank == 0) std::cout << "\n--- GMRES Variants ---\n";
    
    std::vector<std::pair<std::string, Preconditioner*>> gmres_preconds = {
        {"None", nullptr},
        {"Jacobi", &jacobi},
        {"ILU(0)", &ilu0}
    };
    
    std::vector<std::pair<std::string, SolverFunc>> gmres_variants = {
        {"GMRES", gmres_wrapper},
        {"CA-GMRES", ca_gmres_wrapper},
        {"Pipe-GMRES", pipelined_gmres_wrapper}
    };
    
    for (const auto& [solver_name, solver_func] : gmres_variants) {
        for (const auto& [precond_name, precond] : gmres_preconds) {
            std::vector<double> x(local_n, 0.0);
            int iters = 0;
            double final_norm = 0.0;
            
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            int status = solver_func(A, b, x, max_iter, tol, MPI_COMM_WORLD, 
                                    precond, &iters, &final_norm);
            double t1 = MPI_Wtime();
            
            print_result(rank, solver_name, precond_name, iters, final_norm, 
                        t1 - t0, status != 0);
        }
    }

    // Test 4: BiCGStab
    if (rank == 0) std::cout << "\n--- BiCGStab ---\n";
    
    {
        std::vector<double> x(local_n, 0.0);
        int iters = 0;
        double final_norm = 0.0;
        
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        int status = bicgstab_wrapper(A, b, x, max_iter, tol, MPI_COMM_WORLD, 
                                     &jacobi, &iters, &final_norm);
        double t1 = MPI_Wtime();
        
        print_result(rank, "BiCGStab", "Jacobi", iters, final_norm, t1 - t0, status != 0);
    }

    // ============================================================
    // Summary
    // ============================================================
    
    if (rank == 0) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "Benchmark Complete\n";
        std::cout << std::string(80, '=') << "\n\n";
        
        std::cout << "Algorithm Characteristics:\n";
        std::cout << "  Standard CG:      ~2 Allreduce/iteration\n";
        std::cout << "  CA-CG (s=5):      ~2 Allreduce/" << 5 << " iterations\n";
        std::cout << "  Standard GMRES:   ~2+ Allreduce/iteration\n";
        std::cout << "  CA-GMRES (s=5):   ~2 Allreduce/" << 5 << " iterations\n";
        std::cout << "  Pipelined GMRES:  ~1 Allreduce/iteration (overlapped)\n\n";
        
        std::cout << "Preconditioner Characteristics:\n";
        std::cout << "  None:           No communication, poor convergence\n";
        std::cout << "  Jacobi:         No communication, diagonal scaling\n";
        std::cout << "  ILU(0):         No communication (local), best convergence\n";
        std::cout << "  Poly-Neumann:   No communication, moderate convergence\n";
        std::cout << "  Add-Schwarz:    Minimal communication, good convergence\n\n";
        
        std::cout << "Performance Tips:\n";
        std::cout << "  - Use CA methods on high-latency networks (cloud, WAN)\n";
        std::cout << "  - Pair CA methods with communication-free preconditioners\n";
        std::cout << "  - ILU(0)/ASM give best convergence but costly setup\n";
        std::cout << "  - Pipelined methods good for fast interconnects\n\n";
    }

    MPI_Finalize();
    return 0;
}