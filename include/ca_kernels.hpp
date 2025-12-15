#ifndef CA_KERNELS_HPP
#define CA_KERNELS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include "matrix.hpp"
#include "preconditioner.hpp" 


inline bool dense_cholesky_safe(int size, std::vector<double> A, std::vector<double>& R) {
    R.assign(size * size, 0.0);
    double shift = 0.0;
    
    for (int attempt = 0; attempt < 3; ++attempt) {
        bool success = true;
        std::vector<double> L(size * size, 0.0);

        if (shift > 0.0) {
            for(int i=0; i<size; ++i) A[i*size+i] += shift;
        }

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = 0.0;
                for (int k = 0; k < j; ++k) sum += L[i * size + k] * L[j * size + k];
                
                if (i == j) {
                    double val = A[i * size + i] - sum;
                    if (val <= 1e-16) { success = false; break; }
                    L[i * size + j] = std::sqrt(val);
                } else {
                    L[i * size + j] = (1.0 / L[j * size + j]) * (A[i * size + j] - sum);
                }
            }
            if (!success) break;
        }

        if (success) {
            for (int i = 0; i < size; ++i)
                for (int j = 0; j < size; ++j)
                    if (j >= i) R[i * size + j] = L[j * size + i];
            return true;
        }
        shift = (shift == 0.0) ? 1e-10 : shift * 100.0;
    }
    return false;
}

inline void dense_matmul(int size, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    C.assign(size * size, 0.0);
    for(int i=0; i<size; ++i)
        for(int k=0; k<size; ++k)
            for(int j=0; j<size; ++j)
                C[i*size + j] += A[i*size + k] * B[k*size + j];
}

inline void invert_upper_triangular(int size, const std::vector<double>& R, std::vector<double>& R_inv) {
    R_inv.assign(size * size, 0.0);
    for(int i=0; i<size; ++i) {
        if (std::abs(R[i*size+i]) < 1e-16) {
             R_inv.assign(size * size, 0.0);
             for(int k=0; k<size; ++k) R_inv[k*size+k] = 1.0;
             return;
        }
        R_inv[i*size+i] = 1.0 / R[i*size+i];
    }

    for(int i=size-1; i>=0; --i) {
        for(int j=i+1; j<size; ++j) {
            double sum = 0.0;
            for(int k=i+1; k<=j; ++k) sum += R[i*size+k] * R_inv[k*size+j];
            R_inv[i*size+j] = -sum / R[i*size+i];
        }
    }
}

// ***************************************************
// CA Kernels
// ***************************************************

/**
 * @brief Matrix Powers Kernel with Preconditioning
 */
inline void ca_matrix_powers_real(const CSRMatrix& A, 
                                  const std::vector<double>& start_v, 
                                  int s, 
                                  std::vector<double>& V_block, 
                                  Preconditioner* P, 
                                  MPI_Comm comm) {
    int n = A.local_n();
    V_block.resize((s + 1) * n);
    
    for(int i=0; i<n; ++i) V_block[i] = start_v[i];
    
    std::vector<double> z(n); // M^-1 * v
    
    for(int k=0; k<s; ++k) {
        const double* v_curr_ptr = &V_block[k * n];
        double* v_next_ptr = &V_block[(k + 1) * n];
        
        // z = M^{-1} * v_curr
        if (P) {
            std::vector<double> v_temp(v_curr_ptr, v_curr_ptr + n);
            P->apply(v_temp, z);
        } else {
            for(int i=0; i<n; ++i) z[i] = v_curr_ptr[i];
        }

        // A: v_next = A * z
        std::vector<double> y_out(n);
        distributed_matvec(A, z, y_out, comm);
        
        for(int i=0; i<n; ++i) v_next_ptr[i] = y_out[i];
    }
}

/**
 * @brief CholeskyQR
 */
inline void ca_cholesky_qr(std::vector<double>& V, int num_vecs, int local_n, 
                           std::vector<double>& R_out, MPI_Comm comm) {
    
    std::vector<double> R1(num_vecs * num_vecs);
    std::vector<double> R2(num_vecs * num_vecs);
    
    // --- Pass 1 ---
    {
        std::vector<double> G_local(num_vecs * num_vecs, 0.0);
        for(int i=0; i<num_vecs; ++i) {
            for(int j=i; j<num_vecs; ++j) {
                double dot = 0.0;
                for(int k=0; k<local_n; ++k) dot += V[i*local_n + k] * V[j*local_n + k];
                G_local[i*num_vecs + j] = dot;
            }
        }
        for(int i=0; i<num_vecs; ++i) 
            for(int j=0; j<i; ++j) G_local[i*num_vecs+j] = G_local[j*num_vecs+i];

        std::vector<double> G_global(num_vecs * num_vecs);
        MPI_Allreduce(G_local.data(), G_global.data(), num_vecs * num_vecs, MPI_DOUBLE, MPI_SUM, comm);

        if (!dense_cholesky_safe(num_vecs, G_global, R1)) {
            R1.assign(num_vecs*num_vecs, 0.0);
            for(int i=0; i<num_vecs; ++i) R1[i*num_vecs+i] = 1.0;
        }

        std::vector<double> R_inv;
        invert_upper_triangular(num_vecs, R1, R_inv);
        
        std::vector<double> V_old = V;
        for(int k=0; k<local_n; ++k) {
            std::vector<double> row_new(num_vecs, 0.0);
            for(int j=0; j<num_vecs; ++j) {
                for(int i=0; i<num_vecs; ++i) {
                    row_new[j] += V_old[i*local_n + k] * R_inv[i*num_vecs + j];
                }
            }
            for(int j=0; j<num_vecs; ++j) V[j*local_n + k] = row_new[j];
        }
    }

    // --- Pass 2 ---
    {
        std::vector<double> G_local(num_vecs * num_vecs, 0.0);
        for(int i=0; i<num_vecs; ++i) {
            for(int j=i; j<num_vecs; ++j) {
                double dot = 0.0;
                for(int k=0; k<local_n; ++k) dot += V[i*local_n + k] * V[j*local_n + k];
                G_local[i*num_vecs + j] = dot;
            }
        }
        for(int i=0; i<num_vecs; ++i) 
            for(int j=0; j<i; ++j) G_local[i*num_vecs+j] = G_local[j*num_vecs+i];

        std::vector<double> G_global(num_vecs * num_vecs);
        MPI_Allreduce(G_local.data(), G_global.data(), num_vecs * num_vecs, MPI_DOUBLE, MPI_SUM, comm);

        if (!dense_cholesky_safe(num_vecs, G_global, R2)) {
            R2.assign(num_vecs*num_vecs, 0.0);
            for(int i=0; i<num_vecs; ++i) R2[i*num_vecs+i] = 1.0;
        }

        std::vector<double> R_inv;
        invert_upper_triangular(num_vecs, R2, R_inv);

        std::vector<double> V_old = V;
        for(int k=0; k<local_n; ++k) {
            std::vector<double> row_new(num_vecs, 0.0);
            for(int j=0; j<num_vecs; ++j) {
                for(int i=0; i<num_vecs; ++i) {
                    row_new[j] += V_old[i*local_n + k] * R_inv[i*num_vecs + j];
                }
            }
            for(int j=0; j<num_vecs; ++j) V[j*local_n + k] = row_new[j];
        }
    }

    dense_matmul(num_vecs, R2, R1, R_out);
}

#endif // CA_KERNELS_HPP