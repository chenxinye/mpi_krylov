/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 * 
 * Optimized CA-GMRES with efficient communication pattern
 */ 

#include <vector>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include "matrix.hpp"
#include "preconditioner.hpp"
#include "cagmres.hpp"
#include "ca_kernels.hpp"

int cagmres_solve(const CSRMatrix& A, const std::vector<double>& b, std::vector<double>& x,
                  int restart, int s, int maxit, double tol,
                  MPI_Comm comm, Preconditioner* P, int* iters, double* final_res) {
    
    if (s > 5) s = 5; 
    if (s < 1) s = 1;

    int n = A.local_n();
    int rank;
    MPI_Comm_rank(comm, &rank);

    double norm_b = global_norm(b, comm);
    if (norm_b == 0.0) norm_b = 1.0;
    double stop_tol = tol * norm_b;

    // Use optimized matvec if available
    std::vector<double> Ax(n);
    if (A.halo_initialized) {
        distributed_matvec_optimized(A, x, Ax, comm);
    } else {
        distributed_matvec(A, x, Ax, comm);
    }
    
    std::vector<double> r(n);
    for(int i=0; i<n; ++i) r[i] = b[i] - Ax[i];

    double beta = global_norm(r, comm);
    if (beta < stop_tol) {
        if(iters) *iters = 0;
        if(final_res) *final_res = beta;
        return 0;
    }

    int m = (restart > 0) ? restart : 30;
    m = (m / s) * s; 
    if (m == 0) m = s;

    std::vector<double> V((m + s + 1) * n); 
    std::vector<double> H((m + 1) * m, 0.0);
    std::vector<double> cs(m, 0.0);
    std::vector<double> sn(m, 0.0);
    std::vector<double> g(m + 1, 0.0);

    std::vector<double> V_block; 
    std::vector<double> R_block;

    int k = 0; 
    while (k < maxit) {
        // Restart
        double inv_beta = 1.0 / beta;
        for(int i=0; i<n; ++i) V[i] = r[i] * inv_beta;
        
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        int j = 0; 
        while (j < m && k < maxit) {
            
            // Matrix Powers Kernel
            std::vector<double> start_v(n);
            const double* v_ptr = &V[j * n];
            for(int i=0; i<n; ++i) start_v[i] = v_ptr[i];

            ca_matrix_powers_real(A, start_v, s, V_block, P, comm);

            // Use original efficient Cholesky QR
            ca_cholesky_qr(V_block, s + 1, n, R_block, comm);

            // Copy orthogonalized vectors back to V
            for(int step = 1; step <= s; ++step) {
                if (j + step > m) break; 
                double* dst = &V[(j + step) * n];
                const double* src = &V_block[step * n];
                for(int i=0; i<n; ++i) dst[i] = src[i];
            }

            int current_s = (j + s > m) ? (m - j) : s;
            
            // Compute W = A * M^{-1} * V block (single communication phase)
            std::vector<double> W_temp(current_s * n);
            std::vector<double> z(n); 

            for(int step=0; step < current_s; ++step) {
                std::vector<double> v_in(n), w_out(n);
                for(int i=0; i<n; ++i) v_in[i] = V[(j+step)*n + i];
                
                if(P) P->apply(v_in, z);
                else z = v_in;

                // Use optimized matvec
                if (A.halo_initialized) {
                    distributed_matvec_optimized(A, z, w_out, comm);
                } else {
                    distributed_matvec(A, z, w_out, comm);
                }
                
                for(int i=0; i<n; ++i) W_temp[step*n + i] = w_out[i];
            }

            // Compute Hessenberg matrix: H = V^T * W (single Allreduce)
            int h_rows = j + current_s + 1;
            int h_cols = current_s;
            std::vector<double> H_local(h_rows * h_cols, 0.0);
            
            for(int c=0; c < h_cols; ++c) {
                for(int r_idx=0; r_idx < h_rows; ++r_idx) {
                    double dot = 0.0;
                    for(int i=0; i<n; ++i) {
                        dot += V[r_idx*n + i] * W_temp[c*n + i];
                    }
                    H_local[r_idx*h_cols + c] = dot;
                }
            }
            
            std::vector<double> H_global(h_rows * h_cols);
            MPI_Allreduce(H_local.data(), H_global.data(), h_rows*h_cols, 
                         MPI_DOUBLE, MPI_SUM, comm);

            // Store in full H matrix
            for(int c=0; c < h_cols; ++c) {
                for(int r_idx=0; r_idx < h_rows; ++r_idx) {
                    if (r_idx < m + 1 && (j + c) < m) {
                        H[r_idx * m + (j + c)] = H_global[r_idx * h_cols + c];
                    }
                }
            }

            // Apply Givens rotations to new columns
            for (int step = 0; step < current_s; ++step) {
                int col = j + step;
                
                // Apply previous rotations
                for (int i = 0; i < col; ++i) {
                    double temp = cs[i] * H[i*m + col] + sn[i] * H[(i + 1)*m + col];
                    H[(i + 1)*m + col] = -sn[i] * H[i*m + col] + cs[i] * H[(i + 1)*m + col];
                    H[i*m + col] = temp;
                }
                
                // Compute new rotation
                double h1 = H[col*m + col];
                double h2 = H[(col + 1)*m + col];
                double r_sq = std::sqrt(h1*h1 + h2*h2);
                
                if (r_sq > 1e-16) {
                    cs[col] = h1 / r_sq;
                    sn[col] = h2 / r_sq;
                } else {
                    cs[col] = 1.0;
                    sn[col] = 0.0;
                }

                H[col*m + col] = r_sq;
                H[(col + 1)*m + col] = 0.0;

                // Apply to RHS
                g[col + 1] = -sn[col] * g[col];
                g[col]     = cs[col] * g[col];
            }

            j += current_s;
            k += current_s;

            beta = std::abs(g[j]);
            if (beta < stop_tol) break;
        }

        // Solve triangular system
        int effective_m = j;
        std::vector<double> y(effective_m);
        for (int i = effective_m - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int l = i + 1; l < effective_m; ++l) {
                sum += H[i*m + l] * y[l];
            }
            y[i] = (g[i] - sum) / H[i*m + i];
        }

        // Update solution: x += M^{-1} * V * y
        std::vector<double> u(n, 0.0);
        for (int row = 0; row < n; ++row) {
            double sum = 0.0;
            for (int i = 0; i < effective_m; ++i) {
                sum += V[i*n + row] * y[i];
            }
            u[row] = sum;
        }

        std::vector<double> real_update(n);
        if (P) {
            P->apply(u, real_update);
        } else {
            real_update = u;
        }

        for(int i=0; i<n; ++i) x[i] += real_update[i];

        // Compute residual
        if (A.halo_initialized) {
            distributed_matvec_optimized(A, x, Ax, comm);
        } else {
            distributed_matvec(A, x, Ax, comm);
        }
        
        for(int i=0; i<n; ++i) r[i] = b[i] - Ax[i];
        beta = global_norm(r, comm);
        
        if (beta < stop_tol) break;
    }

    if(iters) *iters = k;
    if(final_res) *final_res = beta;
    return 0;
}