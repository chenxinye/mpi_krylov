/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 * Date: 2025-12-09
 */ 

#include "utils.hpp"
#include <numeric>

double compute_residual_norm(const CSRMatrix &A, const std::vector<double>& x_local, const std::vector<double>& b_local, MPI_Comm comm) {
    std::vector<double> Ax_local;
    distributed_matvec(A, x_local, Ax_local, comm);
    std::vector<double> r_local(A.local_n());
    for (int i = 0; i < A.local_n(); ++i) r_local[i] = b_local[i] - Ax_local[i];
    return global_norm(r_local, comm);
}

long long local_nnz(const CSRMatrix &A) { return A.nnz(); }
long long global_nnz(const CSRMatrix &A, MPI_Comm comm) {
    long long local = local_nnz(A);
    long long global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG, MPI_SUM, comm);
    return global;
}