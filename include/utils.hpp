#ifndef UTILS_HPP
#define UTILS_HPP
#include <vector>
#include "matrix.hpp"
#include <mpi.h>

// compute distributed residual norm ||b - A x||
double compute_residual_norm(const CSRMatrix &A, const std::vector<double>& x_local, const std::vector<double>& b_local, MPI_Comm comm);
// count local and global nnz
long long local_nnz(const CSRMatrix &A);
long long global_nnz(const CSRMatrix &A, MPI_Comm comm);

#endif