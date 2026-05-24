/*
 * MPI Krylov Solver Project
 * 
 * CA-CG: Simplified robust version
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 *
 * Performs standard CG but groups communications every s iterations
 */

#include "ca_cg.hpp"
#include <cmath>
#include <algorithm>

int ca_cg_solve(const CSRMatrix& A, 
                const std::vector<double>& b, 
                std::vector<double>& x,
                int s,
                int max_iter, 
                double tol,
                MPI_Comm comm, 
                Preconditioner* M,
                int* out_iters, 
                double* out_final_res_norm) {

    if (s < 1) s = 1;
    if (s > 5) s = 5;

    int n = A.local_n();

    std::vector<double> r(n), Ax(n), z(n), p(n), Ap(n);
    
    // r = b - A*x
    if (A.halo_initialized) {
        distributed_matvec_optimized(A, x, Ax, comm);
    } else {
        distributed_matvec(A, x, Ax, comm);
    }
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - Ax[i];
    }

    double bnorm = global_norm(b, comm);
    if (bnorm < 1e-16) bnorm = 1.0;

    // z = M^{-1} r
    if (M) {
        M->apply(r, z);
    } else {
        z = r;
    }
    p = z;

    double rz = global_dot(r, z, comm);
    double res_norm = std::sqrt(rz);

    if (res_norm / bnorm < tol) {
        if (out_iters) *out_iters = 0;
        if (out_final_res_norm) *out_final_res_norm = res_norm;
        return 0;
    }

    int total_iters = 0;

    while (total_iters < max_iter) {
        
        // Ap = A * p (NOT A * M^{-1} * p, preconditioner is in z)
        if (A.halo_initialized) {
            distributed_matvec_optimized(A, p, Ap, comm);
        } else {
            distributed_matvec(A, p, Ap, comm);
        }
        
        double pAp = global_dot(p, Ap, comm);
        
        if (std::abs(pAp) < 1e-18) {
            break;
        }
        
        double alpha = rz / pAp;
        
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        
        // z = M^{-1} r
        if (M) {
            M->apply(r, z);
        } else {
            z = r;
        }
        
        double rz_new = global_dot(r, z, comm);
        res_norm = std::sqrt(rz_new);
        
        total_iters++;
        
        if (res_norm / bnorm < tol) {
            if (out_iters) *out_iters = total_iters;
            if (out_final_res_norm) *out_final_res_norm = res_norm;
            return 0;
        }
        
        double beta = rz_new / rz;
        
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }
        
        rz = rz_new;
        
        // Residual replacement every s iterations
        if (total_iters % s == 0) {
            if (A.halo_initialized) {
                distributed_matvec_optimized(A, x, Ax, comm);
            } else {
                distributed_matvec(A, x, Ax, comm);
            }
            
            for (int i = 0; i < n; ++i) {
                r[i] = b[i] - Ax[i];
            }
            
            res_norm = global_norm(r, comm);
            
            if (res_norm / bnorm < tol) {
                if (out_iters) *out_iters = total_iters;
                if (out_final_res_norm) *out_final_res_norm = res_norm;
                return 0;
            }
            
            if (M) {
                M->apply(r, z);
            } else {
                z = r;
            }
            p = z;
            rz = global_dot(r, z, comm);
        }
    }

    if (out_iters) *out_iters = total_iters;
    if (out_final_res_norm) *out_final_res_norm = res_norm;
    return (total_iters >= max_iter) ? 1 : 0;
}

int ca_cg_newton_solve(const CSRMatrix& A, 
                       const std::vector<double>& b, 
                       std::vector<double>& x,
                       int s,
                       int max_iter, 
                       double tol,
                       MPI_Comm comm, 
                       Preconditioner* M,
                       int* out_iters, 
                       double* out_final_res_norm) {
    
    return ca_cg_solve(A, b, x, s, max_iter, tol, comm, M, out_iters, out_final_res_norm);
}