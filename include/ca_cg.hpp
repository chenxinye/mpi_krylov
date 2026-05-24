#ifndef CA_CG_HPP
#define CA_CG_HPP

#include <vector>
#include <mpi.h>
#include "matrix.hpp"
#include "preconditioner.hpp"

/**
 * @brief Communication-Avoiding Conjugate Gradient (CA-CG)
 * 
 * s-step variant that reduces global synchronizations from 2 per iteration
 * to 2 per s iterations using matrix powers kernel and block orthogonalization.
 * 
 * @param A Distributed CSR matrix (must be symmetric positive definite)
 * @param b Right-hand side vector
 * @param x Solution vector (input: initial guess, output: solution)
 * @param s Basis step size (number of iterations before communication)
 * @param max_iter Maximum number of iterations
 * @param tol Convergence tolerance
 * @param comm MPI communicator
 * @param M Preconditioner (can be nullptr)
 * @param out_iters Output: actual number of iterations
 * @param out_final_res_norm Output: final residual norm
 * @return 0 on success, 1 if max iterations reached
 */
int ca_cg_solve(const CSRMatrix& A, 
                const std::vector<double>& b, 
                std::vector<double>& x,
                int s,
                int max_iter, 
                double tol,
                MPI_Comm comm, 
                Preconditioner* M = nullptr,
                int* out_iters = nullptr, 
                double* out_final_res_norm = nullptr);

/**
 * @brief CA-CG with Newton basis (more stable than monomial basis)
 */
int ca_cg_newton_solve(const CSRMatrix& A, 
                       const std::vector<double>& b, 
                       std::vector<double>& x,
                       int s,
                       int max_iter, 
                       double tol,
                       MPI_Comm comm, 
                       Preconditioner* M = nullptr,
                       int* out_iters = nullptr, 
                       double* out_final_res_norm = nullptr);

#endif