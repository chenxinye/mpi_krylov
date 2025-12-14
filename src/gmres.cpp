/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 

#include "gmres.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

void distributed_matvec(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y, MPI_Comm comm);
double global_norm(const std::vector<double>& x, MPI_Comm comm);
double global_dot(const std::vector<double>& x, const std::vector<double>& y, MPI_Comm comm);

// To update solution: x = x + M^{-1} * (V * y)
// If M is null, x = x + (V * y)
void update_solution(const std::vector<std::vector<double>>& V, 
                     const std::vector<double>& y, 
                     std::vector<double>& x, 
                     int n, int m, 
                     Preconditioner* M) {
    
    // Compute linear combination w = V * y
    std::vector<double> w(n, 0.0);
    for (int k = 0; k <= m; ++k) {
        for (int i = 0; i < n; ++i) {
            w[i] += V[k][i] * y[k];
        }
    }

    // Apply Preconditioner: u = M^{-1} * w
    std::vector<double> u(n);
    if (M) {
        M->apply(w, u);
    } else {
        u = w;
    }

    // 3. Update x: x = x + u
    for (int i = 0; i < n; ++i) {
        x[i] += u[i];
    }
}

int gmres_solve(const CSRMatrix& A, const std::vector<double>& b_local, std::vector<double>& x_local,
                int restart, int max_iter, double tol, MPI_Comm comm, Preconditioner* M,
                int *out_iters, double *out_final_res_norm) {
    
    // Use A.nrows if A is the struct from previous context, or A.local_n() if using a method.
    int n = A.nrows; 
    
    std::vector<double> r(n), Av(n), tmp(n); // tmp used for preconditioning buffer
    
    // Initial Residual: r = b - A*x
    distributed_matvec(A, x_local, Av, comm);
    for (int i = 0; i < n; ++i) r[i] = b_local[i] - Av[i];
    
    double bnorm = global_norm(b_local, comm); 
    if (bnorm < 1e-16) bnorm = 1.0;

    int total_iters = 0;
    
    // Storage for Arnoldi Basis (V) and Hessenberg Matrix (H)
    std::vector<std::vector<double>> V(restart + 1, std::vector<double>(n));
    std::vector<std::vector<double>> H(restart + 1, std::vector<double>(restart));
    std::vector<double> cs(restart, 0.0), sn(restart, 0.0), e1(restart + 1, 0.0);

    double beta = global_norm(r, comm);
    
    // Check initial convergence
    if (beta / bnorm < tol) {
        if (out_iters) *out_iters = 0;
        if (out_final_res_norm) *out_final_res_norm = beta;
        return 0;
    }

    while (total_iters < max_iter) {
        //Restart Setup
        e1.assign(restart + 1, 0.0);
        e1[0] = beta;
        
        // Clear H (optional but safe)
        for (auto& row : H) std::fill(row.begin(), row.end(), 0.0);
        
        // V[0] = r / beta
        for (int i = 0; i < n; ++i) V[0][i] = r[i] / beta;

        int m = 0;
        bool converged_inner = false;

        //Arnoldi Process
        for (; m < restart && total_iters < max_iter; ++m) {
            
            // Right Preconditioning: w = A * (M^{-1} * V[m])
            if (M) {
                M->apply(V[m], tmp); // tmp = M^{-1} * V[m]
                distributed_matvec(A, tmp, Av, comm); // Av = A * tmp
            } else {
                distributed_matvec(A, V[m], Av, comm);
            }

            // Modified Gram-Schmidt Orthogonalization
            for (int j = 0; j <= m; ++j) {
                H[j][m] = global_dot(Av, V[j], comm);
                for (int i = 0; i < n; ++i) Av[i] -= H[j][m] * V[j][i];
            }
            
            H[m+1][m] = global_norm(Av, comm);
            
            // Happy Breakdown Check
            if (H[m+1][m] < 1e-16) {
                // Exact subspace found. Proceed to update solution.
                // We stop the Arnoldi loop but perform the update logic below.
                m++; // Increment m to include this step in update
                break; 
            }
            
            // Normalize next basis vector
            for (int i = 0; i < n; ++i) V[m+1][i] = Av[i] / H[m+1][m];

            // Apply Previous Givens Rotations
            for (int i = 0; i < m; ++i) {
                double temp = cs[i] * H[i][m] + sn[i] * H[i+1][m];
                H[i+1][m] = -sn[i] * H[i][m] + cs[i] * H[i+1][m];
                H[i][m] = temp;
            }

            // Generate New Rotation
            double rho_val = std::hypot(H[m][m], H[m+1][m]);
            if (rho_val < 1e-16) { 
                cs[m] = 1.0; sn[m] = 0.0; 
            } else { 
                cs[m] = H[m][m] / rho_val; 
                sn[m] = H[m+1][m] / rho_val; 
            }

            // Apply rotation to H and RHS (e1)
            H[m][m] = cs[m] * H[m][m] + sn[m] * H[m+1][m];
            H[m+1][m] = 0.0;
            
            double temp_e = cs[m] * e1[m] + sn[m] * e1[m+1];
            e1[m+1] = -sn[m] * e1[m] + cs[m] * e1[m+1];
            e1[m] = temp_e;

            // Check Convergence
            ++total_iters;
            double rel_res = std::abs(e1[m+1]) / bnorm;
            
            if (rel_res < tol) {
                converged_inner = true;
                break; // Break inner loop, proceed to update x
            }
        }

        //Solution Update Step
        // Solve upper triangular system H y = e1
        int m_used = converged_inner ? m : m - 1; // Adjust index based on how loop exited
        if (m_used < 0) m_used = 0;

        std::vector<double> y(m_used + 1, 0.0);
        for (int i = m_used; i >= 0; --i) {
            double s = e1[i];
            for (int j = i + 1; j <= m_used; ++j) s -= H[i][j] * y[j];
            y[i] = s / H[i][i];
        }

        // Update x: x = x + M^{-1} * (V * y)
        update_solution(V, y, x_local, n, m_used, M);

        // Check if we converged
        if (converged_inner) {
            if (out_iters) *out_iters = total_iters;
            if (out_final_res_norm) *out_final_res_norm = std::abs(e1[m_used+1]);
            return 0;
        }

        //Restart: Recompute Residual Explicitly
        // r = b - A*x to remove accumulated floating point errors
        distributed_matvec(A, x_local, Av, comm);
        for (int i = 0; i < n; ++i) r[i] = b_local[i] - Av[i];
        beta = global_norm(r, comm);
        
        // Final check before next restart
        if (beta / bnorm < tol) {
            if (out_iters) *out_iters = total_iters;
            if (out_final_res_norm) *out_final_res_norm = beta;
            return 0;
        }
    }

    // Max iterations reached
    if (out_iters) *out_iters = total_iters;
    if (out_final_res_norm) *out_final_res_norm = beta;
    return 1;
}