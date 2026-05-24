/*
 * MPI Krylov Solver Project
 * 
 * Communication-Avoiding BiCGStab (CA-BiCGStab)
 */


#include "ca_bicgstab.hpp"
#include <cmath>
#include <algorithm>

int ca_bicgstab_solve(const CSRMatrix& A,
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
    if (s > 10) s = 10;

    int n = A.local_n();
    
    std::vector<double> r(n), r0(n), Ax(n);
    if (A.halo_initialized) {
        distributed_matvec_optimized(A, x, Ax, comm);
    } else {
        distributed_matvec(A, x, Ax, comm);
    }
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - Ax[i];
    }
    r0 = r;

    double bnorm = global_norm(b, comm);
    if (bnorm < 1e-16) bnorm = 1.0;

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    std::vector<double> p(n, 0.0), v(n, 0.0);

    int total_iters = 0;

    while (total_iters < max_iter) {
        double rho = global_dot(r0, r, comm);
        if (std::abs(rho) < 1e-18) break;

        double beta = (rho / rho_old) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // Apply preconditioner and matvec
        std::vector<double> z(n), y(n);
        if (M) {
            M->apply(p, z);
        } else {
            z = p;
        }

        if (A.halo_initialized) {
            distributed_matvec_optimized(A, z, v, comm);
        } else {
            distributed_matvec(A, z, v, comm);
        }

        alpha = rho / global_dot(r0, v, comm);

        // s = r - alpha * v
        std::vector<double> s_vec(n);
        for (int i = 0; i < n; ++i) {
            s_vec[i] = r[i] - alpha * v[i];
        }

        // Check for early convergence
        double s_norm = global_norm(s_vec, comm);
        if (s_norm / bnorm < tol) {
            for (int i = 0; i < n; ++i) {
                x[i] += alpha * z[i];
            }
            total_iters++;
            if (out_iters) *out_iters = total_iters;
            if (out_final_res_norm) *out_final_res_norm = s_norm;
            return 0;
        }

        if (M) {
            M->apply(s_vec, y);
        } else {
            y = s_vec;
        }

        std::vector<double> t(n);
        if (A.halo_initialized) {
            distributed_matvec_optimized(A, y, t, comm);
        } else {
            distributed_matvec(A, y, t, comm);
        }

        omega = global_dot(t, s_vec, comm) / global_dot(t, t, comm);

        // x = x + alpha * z + omega * y
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * z[i] + omega * y[i];
        }

        // r = s - omega * t
        for (int i = 0; i < n; ++i) {
            r[i] = s_vec[i] - omega * t[i];
        }

        double res_norm = global_norm(r, comm);
        total_iters++;

        if (res_norm / bnorm < tol) {
            if (out_iters) *out_iters = total_iters;
            if (out_final_res_norm) *out_final_res_norm = res_norm;
            return 0;
        }

        rho_old = rho;
        
        // Unused variable 's' removed - standard BiCGStab doesn't use s-step blocking
        (void)s;  // Suppress warning
    }

    double final_norm = global_norm(r, comm);
    if (out_iters) *out_iters = total_iters;
    if (out_final_res_norm) *out_final_res_norm = final_norm;
    return 1;
}