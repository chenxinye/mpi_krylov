/*
 * MPI Krylov Solver Project
*
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 * 
 * Additive Schwarz Method (ASM) Preconditioner
 * Parallel-friendly: each rank independently solves its local subdomain
 */

#include "additive_schwarz.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

AdditiveSchwarzPrecond::AdditiveSchwarzPrecond(const CSRMatrix& A, int overlap_size, MPI_Comm mpi_comm)
    : factorized(false) {
    
    // Suppress unused parameter warnings
    (void)overlap_size;  // Reserved for future overlap implementation
    (void)mpi_comm;      // Reserved for future communication in overlap
    
    local_size = A.nrows;
    
    // Extract local subdomain as dense matrix
    local_A_dense.resize(local_size * local_size, 0.0);
    
    for (int i = 0; i < A.nrows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int col = A.col_idx[j];
            int local_col = col - A.row_offset;
            
            // Only store strictly local entries
            if (local_col >= 0 && local_col < local_size) {
                local_A_dense[i * local_size + local_col] = A.values[j];
            }
        }
    }
    
    factorize_local_subdomain();
}

void AdditiveSchwarzPrecond::factorize_local_subdomain() {
    // Simple LU factorization with partial pivoting
    
    int n = local_size;
    local_A_inv = local_A_dense;  // Copy for in-place factorization
    pivot.resize(n);
    
    for (int i = 0; i < n; ++i) pivot[i] = i;
    
    for (int k = 0; k < n; ++k) {
        // Find pivot
        int pivot_row = k;
        double max_val = std::abs(local_A_inv[k * n + k]);
        
        for (int i = k + 1; i < n; ++i) {
            double val = std::abs(local_A_inv[i * n + k]);
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }
        
        if (max_val < 1e-14) {
            // Singular matrix, add diagonal perturbation
            local_A_inv[k * n + k] += 1e-8;
        }
        
        // Swap rows
        if (pivot_row != k) {
            std::swap(pivot[k], pivot[pivot_row]);
            for (int j = 0; j < n; ++j) {
                std::swap(local_A_inv[k * n + j], local_A_inv[pivot_row * n + j]);
            }
        }
        
        // Eliminate
        for (int i = k + 1; i < n; ++i) {
            double factor = local_A_inv[i * n + k] / local_A_inv[k * n + k];
            local_A_inv[i * n + k] = factor;  // Store L
            
            for (int j = k + 1; j < n; ++j) {
                local_A_inv[i * n + j] -= factor * local_A_inv[k * n + j];
            }
        }
    }
    
    factorized = true;
}

void AdditiveSchwarzPrecond::apply(const std::vector<double>& r_local, std::vector<double>& z_local) {
    if (!factorized) {
        z_local = r_local;
        return;
    }
    
    int n = local_size;
    z_local = r_local;
    
    // Apply row permutation
    std::vector<double> tmp(n);
    for (int i = 0; i < n; ++i) {
        tmp[i] = z_local[pivot[i]];
    }
    z_local = tmp;
    
    // Forward substitution (L y = P b)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            z_local[i] -= local_A_inv[i * n + j] * z_local[j];
        }
    }
    
    // Backward substitution (U x = y)
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            z_local[i] -= local_A_inv[i * n + j] * z_local[j];
        }
        z_local[i] /= local_A_inv[i * n + i];
    }
}