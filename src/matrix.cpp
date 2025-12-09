/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 

#include "matrix.hpp"
#include <numeric>
#include <cassert>

void distributed_matvec(const CSRMatrix &A, const std::vector<double> &x_local, std::vector<double> &y_local, MPI_Comm comm) {
    int rank, size; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);
    // gather local sizes
    int local_n = (int)x_local.size();
    std::vector<int> recvcounts(size), displs(size);
    MPI_Allgather(&local_n, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + recvcounts[i-1];
    int total = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
    std::vector<double> x_global(total);
    MPI_Allgatherv(x_local.data(), local_n, MPI_DOUBLE, x_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, comm);

    // basic local multiply: y_local = A_local * x_global
    y_local.assign(A.local_n(), 0.0);
    for (int i = 0; i < A.local_n(); ++i) {
        double s = 0.0;
        for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; ++jj) {
            int col = A.col_idx[jj];
            s += A.values[jj] * x_global[col];
        }
        y_local[i] = s;
    }
}

double global_dot(const std::vector<double>& a_local, const std::vector<double>& b_local, MPI_Comm comm) {
    assert(a_local.size() == b_local.size());
    double local = 0.0;
    for (size_t i = 0; i < a_local.size(); ++i) local += a_local[i] * b_local[i];
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

double global_norm(const std::vector<double>& a_local, MPI_Comm comm) {
    return std::sqrt(global_dot(a_local, a_local, comm));
}