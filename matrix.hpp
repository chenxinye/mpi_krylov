#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <mpi.h>

struct CSRMatrix {
    int nrows = 0;              // local number of rows
    int ncols = 0;              // global number of columns (for simplicity assume square global)
    int row_offset = 0;         // global index of first local row
    std::vector<int> row_ptr;   // size nrows+1
    std::vector<int> col_idx;   // global column indices
    std::vector<double> values; // nonzero values

    int local_n() const { return nrows; }
    int global_n() const { return ncols; }
    long long nnz() const { return (long long)values.size(); }
};

// simple distributed matvec that gathers full x across ranks (robust, easy)
void distributed_matvec(const CSRMatrix &A, const std::vector<double> &x_local, std::vector<double> &y_local, MPI_Comm comm);

// global dot product and norm helpers
double global_dot(const std::vector<double>& a_local, const std::vector<double>& b_local, MPI_Comm comm);
double global_norm(const std::vector<double>& a_local, MPI_Comm comm);

#endif