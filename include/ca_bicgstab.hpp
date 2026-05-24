#ifndef CA_BICGSTAB_HPP
#define CA_BICGSTAB_HPP

#include <vector>
#include <mpi.h>
#include "matrix.hpp"
#include "preconditioner.hpp"

/**
 * @brief Communication-Avoiding BiCGStab (CA-BiCGStab)
 * 
 * s-step variant for non-symmetric systems
 * 
 * @param A Distributed CSR matrix
 * @param b Right-hand side vector
 * @param x Solution vector
 * @param s Basis step size
 * @param max_iter Maximum iterations
 * @param tol Convergence tolerance
 * @param comm MPI communicator
 * @param M Preconditioner
 * @param out_iters Output iterations
 * @param out_final_res_norm Output residual norm
 */
int ca_bicgstab_solve(const CSRMatrix& A,
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