/*
 * MPI Krylov Solver Project
 * 
 * Jacobi Preconditioner - FIXED
 */

#include "jacobi.hpp"
#include <cmath>
#include <algorithm>

JacobiPrecond::JacobiPrecond(const CSRMatrix& A) {
    int n = A.nrows;
    invdiag_local.resize(n);
    
    // Extract diagonal elements
    for (int i = 0; i < n; ++i) {
        int global_i = A.row_offset + i;
        double diag_val = 0.0;
        bool found = false;
        
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_idx[j] == global_i) {
                diag_val = A.values[j];
                found = true;
                break;
            }
        }
        
        // Safety check: ensure diagonal is nonzero
        if (!found || std::abs(diag_val) < 1e-14) {
            invdiag_local[i] = 1.0;  // Identity if diagonal missing/zero
        } else {
            invdiag_local[i] = 1.0 / diag_val;
        }
    }
}

void JacobiPrecond::apply(const std::vector<double>& r_local, std::vector<double>& z_local) {
    int n = r_local.size();
    z_local.resize(n);
    
    for (int i = 0; i < n; ++i) {
        z_local[i] = invdiag_local[i] * r_local[i];
    }
}