#ifndef PIPELINED_GMRES_HPP
#define PIPELINED_GMRES_HPP

#include <vector>
#include <mpi.h>
#include "matrix.hpp"
#include "preconditioner.hpp"

/**
 * @brief Pipelined GMRES
 * 
 * Reduces global synchronization points by overlapping communication
 * with computation. Performs 1 Allreduce per iteration instead of 2+.
 * 
 * Reference:
 * - Ghysels, P., & Vanroose, W. (2014). Hiding global synchronization 
 *   latency in the preconditioned Conjugate Gradient algorithm.
 */
int pipelined_gmres_solve(const CSRMatrix& A,
                          const std::vector<double>& b_local,
                          std::vector<double>& x_local,
                          int restart,
                          int max_iter,
                          double tol,
                          MPI_Comm comm,
                          Preconditioner* M = nullptr,
                          int* out_iters = nullptr,
                          double* out_final_res_norm = nullptr);

#endif