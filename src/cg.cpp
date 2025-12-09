/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 * Date: 2025-12-09
 */ 

#include "cg.hpp"

int cg_solve(const CSRMatrix& A, const std::vector<double>& b_local, std::vector<double>& x_local,
             int max_iter, double tol, MPI_Comm comm, Preconditioner* M,
             int *out_iters, double *out_final_res_norm) {
    int n = A.local_n();
    std::vector<double> r(n), z(n), p(n), Ap(n);
    // r = b - A*x
    distributed_matvec(A, x_local, Ap, comm);
    for (int i = 0; i < n; ++i) r[i] = b_local[i] - Ap[i];
    if (M) M->apply(r, z); else z = r;
    p = z;
    double rz_old = global_dot(r, z, comm);
    double bnorm = global_norm(b_local, comm);
    if (bnorm < 1e-16) bnorm = 1.0;

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        distributed_matvec(A, p, Ap, comm);
        double denom = global_dot(p, Ap, comm);
        if (std::abs(denom) < 1e-18) break;
        double alpha = rz_old / denom;
        for (int i = 0; i < n; ++i) x_local[i] += alpha * p[i];
        for (int i = 0; i < n; ++i) r[i] -= alpha * Ap[i];
        if (M) M->apply(r, z); else z = r;
        double rz_new = global_dot(r, z, comm);
        double rel = std::sqrt(rz_new) / bnorm;
        if (rel < tol) { ++iter; break; }
        double beta = rz_new / rz_old;
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
        rz_old = rz_new;
    }
    double final_norm = global_norm(r, comm);
    if (out_iters) *out_iters = iter;
    if (out_final_res_norm) *out_final_res_norm = final_norm;
    return (iter >= max_iter) ? 1 : 0;
}
