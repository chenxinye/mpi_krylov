/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 * Date: 2025-12-09
 */ 

#include "jacobi.hpp"

JacobiPrecond::JacobiPrecond(const CSRMatrix &A) {
    invdiag_local.resize(A.nrows, 0.0);
    for (int i = 0; i < A.nrows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j) {
            if (A.col_idx[j] == i) invdiag_local[i] = 1.0 / A.values[j];
        }
    }
}

void JacobiPrecond::apply(const std::vector<double>& r_local, std::vector<double>& z_local) {
    int n = r_local.size();
    z_local.resize(n);
    for (int i = 0; i < n; ++i) z_local[i] = invdiag_local[i] * r_local[i];
}
