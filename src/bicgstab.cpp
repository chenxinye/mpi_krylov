/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 

#include "bicgstab.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm> // For std::max
#include <mpi.h>


void distributed_matvec(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y, MPI_Comm comm);
double global_norm(const std::vector<double>& x, MPI_Comm comm);
double global_dot(const std::vector<double>& x, const std::vector<double>& y, MPI_Comm comm);

int bicgstab_solve(const CSRMatrix& A,
                   const std::vector<double>& b_local,
                   std::vector<double>& x_local,
                   int max_iter, double tol,
                   MPI_Comm comm,
                   Preconditioner* M,
                   int* out_iters,
                   double* out_final_res_norm) {
    
    int n = A.nrows;
    // Allocation
    // r: residual, r0: shadow residual, p: search direction
    // v: A * M^-1 * p, s: intermediate residual, t: A * M^-1 * s
    // z: M^-1 * p, y: M^-1 * s
    std::vector<double> r(n), r0(n), p(n), v(n), s(n), t(n), z(n), y(n);
    
    double epsilon = std::numeric_limits<double>::epsilon() * 1e3;

    // Initialize: r = b - A*x
    distributed_matvec(A, x_local, v, comm);
    for (int i = 0; i < n; ++i) r[i] = b_local[i] - v[i];
    r0 = r; // Shadow residual choice: r0 = r

    // Compute Norms for convergence/breakdown checks
    double bnorm = global_norm(b_local, comm);
    double anorm = 0.0;
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            row_sum += std::abs(A.values[j]);
        }
        anorm = std::max(anorm, row_sum);
    }
    
    double global_anorm;
    MPI_Allreduce(&anorm, &global_anorm, 1, MPI_DOUBLE, MPI_MAX, comm);
    
    // Safety for zero vectors
    if (bnorm < epsilon) bnorm = 1.0;
    if (global_anorm < epsilon) global_anorm = 1.0;

    // Check initial residual
    double r_norm = global_norm(r, comm);
    if (r_norm / bnorm < tol) {
        if (out_iters) *out_iters = 0;
        if (out_final_res_norm) *out_final_res_norm = r_norm;
        return 0;
    }

    // BiCGSTAB Variables
    double rho = 1.0, rho_old = 1.0;
    double alpha = 1.0, omega = 1.0, beta = 0.0;

    // 3. Main Loop
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        
        rho = global_dot(r0, r, comm);
        
        if (std::abs(rho) < epsilon * bnorm * global_anorm) {
            // Breakdown: rho approx 0
            break; 
        }

        if (iter == 0) {
            p = r;
        } else {
            beta = (rho / rho_old) * (alpha / omega);
            if (!std::isfinite(beta)) break;

            // p = r + beta * (p - omega * v)
            for (int i = 0; i < n; ++i) {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }

        // Preconditioning  z = M^-1 * p
        if (M) {
            M->apply(p, z);
        } else {
            z = p;
        }

        // v = A * z
        distributed_matvec(A, z, v, comm);

        double r0v = global_dot(r0, v, comm);
        if (std::abs(r0v) < epsilon) break; // Breakdown

        alpha = rho / r0v;
        if (!std::isfinite(alpha)) break;

        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];

        // Early convergence check on s
        double s_norm = global_norm(s, comm);
        if (s_norm / bnorm < tol) {
            // Update x and finish: x = x + alpha * z
            for (int i = 0; i < n; ++i) x_local[i] += alpha * z[i];
            r = s; // Final residual
            iter++; 
            break;
        }

        // Preconditioning (Residual)
        // y = M^-1 * s
        if (M) {
            M->apply(s, y);
        } else {
            y = s;
        }

        // t = A * y
        distributed_matvec(A, y, t, comm);

        double ts = global_dot(t, s, comm);
        double tt = global_dot(t, t, comm);
        
        if (std::abs(tt) < epsilon) {
            // Breakdown or lucky convergence? 
            // Assuming breakdown if s was not small enough previously.
            break; 
        }
        
        omega = ts / tt;
        if (!std::isfinite(omega) || std::abs(omega) < epsilon) break;

        // x = x + alpha * z + omega * y
        // r = s - omega * t
        for (int i = 0; i < n; ++i) {
            x_local[i] += alpha * z[i] + omega * y[i];
            r[i] = s[i] - omega * t[i];
        }

        // Convergence Check
        r_norm = global_norm(r, comm);
        if (r_norm / bnorm < tol) {
            iter++;
            break;
        }
        
        if (!std::isfinite(r_norm)) break;

        // Prepare for next iteration
        rho_old = rho;
    }

    // Recalculate exact residual to ensure accuracy (optional but recommended)
    distributed_matvec(A, x_local, v, comm);
    for (int i = 0; i < n; ++i) r[i] = b_local[i] - v[i];
    double final_norm = global_norm(r, comm);

    if (out_iters) *out_iters = iter;
    if (out_final_res_norm) *out_final_res_norm = final_norm;

    // Return 0 if converged, 1 otherwise
    return (final_norm / bnorm < tol) ? 0 : 1;
}