/*
 * MPI Krylov Solver Project
 * 
 * Pipelined GMRES - Reduced synchronization variant
 */

#include "pipelined_gmres.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

int pipelined_gmres_solve(const CSRMatrix& A,
                          const std::vector<double>& b_local,
                          std::vector<double>& x_local,
                          int restart,
                          int max_iter,
                          double tol,
                          MPI_Comm comm,
                          Preconditioner* M,
                          int* out_iters,
                          double* out_final_res_norm) {
    
    int n = A.nrows;
    std::vector<double> r(n), Av(n), tmp(n);
    
    distributed_matvec(A, x_local, Av, comm);
    for (int i = 0; i < n; ++i) r[i] = b_local[i] - Av[i];
    
    double bnorm = global_norm(b_local, comm);
    if (bnorm < 1e-16) bnorm = 1.0;

    int total_iters = 0;
    
    std::vector<std::vector<double>> V(restart + 1, std::vector<double>(n));
    std::vector<std::vector<double>> H(restart + 1, std::vector<double>(restart));
    std::vector<double> cs(restart, 0.0), sn(restart, 0.0), e1(restart + 1, 0.0);

    double beta = global_norm(r, comm);
    
    if (beta / bnorm < tol) {
        if (out_iters) *out_iters = 0;
        if (out_final_res_norm) *out_final_res_norm = beta;
        return 0;
    }

    while (total_iters < max_iter) {
        e1.assign(restart + 1, 0.0);
        e1[0] = beta;
        
        for (auto& row : H) std::fill(row.begin(), row.end(), 0.0);
        
        for (int i = 0; i < n; ++i) V[0][i] = r[i] / beta;

        int m = 0;
        bool converged_inner = false;

        for (; m < restart && total_iters < max_iter; ++m) {
            // Pipelined computation: overlap matvec and dot products
            if (M) {
                M->apply(V[m], tmp);
                distributed_matvec(A, tmp, Av, comm);
            } else {
                distributed_matvec(A, V[m], Av, comm);
            }

            // Compute all inner products in one Allreduce
            std::vector<double> local_dots(m + 2, 0.0);
            for (int j = 0; j <= m; ++j) {
                for (int i = 0; i < n; ++i) {
                    local_dots[j] += Av[i] * V[j][i];
                }
            }
            // Also compute norm of Av
            for (int i = 0; i < n; ++i) {
                local_dots[m + 1] += Av[i] * Av[i];
            }
            
            std::vector<double> global_dots(m + 2);
            MPI_Allreduce(local_dots.data(), global_dots.data(), m + 2, MPI_DOUBLE, MPI_SUM, comm);
            
            // Extract results
            for (int j = 0; j <= m; ++j) {
                H[j][m] = global_dots[j];
            }
            H[m + 1][m] = std::sqrt(global_dots[m + 1]);
            
            // Modified Gram-Schmidt orthogonalization
            for (int j = 0; j <= m; ++j) {
                for (int i = 0; i < n; ++i) {
                    Av[i] -= H[j][m] * V[j][i];
                }
            }
            
            // Recompute norm after orthogonalization
            double norm_local = 0.0;
            for (int i = 0; i < n; ++i) {
                norm_local += Av[i] * Av[i];
            }
            double norm_global = 0.0;
            MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_SUM, comm);
            H[m + 1][m] = std::sqrt(norm_global);
            
            if (H[m + 1][m] < 1e-16) {
                m++;
                break;
            }
            
            for (int i = 0; i < n; ++i) V[m + 1][i] = Av[i] / H[m + 1][m];

            // Apply Givens rotations
            for (int i = 0; i < m; ++i) {
                double temp = cs[i] * H[i][m] + sn[i] * H[i + 1][m];
                H[i + 1][m] = -sn[i] * H[i][m] + cs[i] * H[i + 1][m];
                H[i][m] = temp;
            }

            double rho_val = std::hypot(H[m][m], H[m + 1][m]);
            if (rho_val < 1e-16) {
                cs[m] = 1.0; sn[m] = 0.0;
            } else {
                cs[m] = H[m][m] / rho_val;
                sn[m] = H[m + 1][m] / rho_val;
            }

            H[m][m] = cs[m] * H[m][m] + sn[m] * H[m + 1][m];
            H[m + 1][m] = 0.0;
            
            double temp_e = cs[m] * e1[m] + sn[m] * e1[m + 1];
            e1[m + 1] = -sn[m] * e1[m] + cs[m] * e1[m + 1];
            e1[m] = temp_e;

            ++total_iters;
            double rel_res = std::abs(e1[m + 1]) / bnorm;
            
            if (rel_res < tol) {
                converged_inner = true;
                break;
            }
        }

        int m_used = converged_inner ? m : m - 1;
        if (m_used < 0) m_used = 0;

        std::vector<double> y(m_used + 1, 0.0);
        for (int i = m_used; i >= 0; --i) {
            double s = e1[i];
            for (int j = i + 1; j <= m_used; ++j) s -= H[i][j] * y[j];
            y[i] = s / H[i][i];
        }

        std::vector<double> update(n, 0.0);
        for (int i = 0; i <= m_used; ++i) {
            for (int k = 0; k < n; ++k) {
                update[k] += y[i] * V[i][k];
            }
        }
        
        if (M) {
            M->apply(update, tmp);
            for (int i = 0; i < n; ++i) x_local[i] += tmp[i];
        } else {
            for (int i = 0; i < n; ++i) x_local[i] += update[i];
        }

        if (converged_inner) {
            if (out_iters) *out_iters = total_iters;
            if (out_final_res_norm) *out_final_res_norm = std::abs(e1[m_used + 1]);
            return 0;
        }

        distributed_matvec(A, x_local, Av, comm);
        for (int i = 0; i < n; ++i) r[i] = b_local[i] - Av[i];
        beta = global_norm(r, comm);
        
        if (beta / bnorm < tol) {
            if (out_iters) *out_iters = total_iters;
            if (out_final_res_norm) *out_final_res_norm = beta;
            return 0;
        }
    }

    if (out_iters) *out_iters = total_iters;
    if (out_final_res_norm) *out_final_res_norm = beta;
    return 1;
}