/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 
 
#include "bicgstab.hpp"

int bicgstab_solve(const CSRMatrix& A,
                    const std::vector<double>& b_local,
                    std::vector<double>& x_local,
                    int max_iter, double tol,
                    MPI_Comm comm,
                    Preconditioner* M,
                    int* out_iters,
                    double* out_final_res_norm) {
    int n = A.nrows;
    std::vector<double> r(n), r0(n), p(n), v(n), s(n), t(n), z(n);
    distributed_matvec(A, x_local, v, comm);
    for (int i = 0; i < n; ++i) r[i] = b_local[i] - v[i];
    r0 = r;
    if (M) M->apply(r, z); else z = r;
    p = z;
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double bnorm = global_norm(b_local, comm); if (bnorm < 1e-16) bnorm = 1.0;
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        double rho = global_dot(r0, z, comm);
        if (std::abs(rho) < 1e-18) break;
        double beta = (rho / rho_old) * (alpha / omega);
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * (p[i] - omega * v[i]);
        distributed_matvec(A, p, v, comm);
        double tmp = global_dot(r0, v, comm);
        if (std::abs(tmp) < 1e-18) break;
        alpha = rho / tmp;
        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        double s_norm = global_norm(s, comm);
        if (s_norm / bnorm < tol) {
            for (int i = 0; i < n; ++i) x_local[i] += alpha * p[i];
            ++iter; break;
        }
        distributed_matvec(A, s, t, comm);
        double tdot = global_dot(t, t, comm);
        if (std::abs(tdot) < 1e-18) break;
        omega = global_dot(t, s, comm) / tdot;
        for (int i = 0; i < n; ++i) {
            x_local[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }
        if (global_norm(r, comm) / bnorm < tol) { ++iter; break; }
        if (std::abs(omega) < 1e-18) break;
        if (M) M->apply(r, z); else z = r;
        rho_old = rho;
    }
    double final_norm = global_norm(r, comm);
    if (out_iters) *out_iters = iter;
    if (out_final_res_norm) *out_final_res_norm = final_norm;
    return (iter >= max_iter) ? 1 : 0;
}