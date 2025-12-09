#include "gmres.hpp"

int gmres_solve(const CSRMatrix& A, const std::vector<double>& b_local, std::vector<double>& x_local,
                int restart, int max_iter, double tol, MPI_Comm comm, Preconditioner* M,
                int *out_iters, double *out_final_res_norm) {
    int n = A.local_n();
    std::vector<double> r(n), Av(n);
    distributed_matvec(A, x_local, Av, comm);
    for (int i = 0; i < n; ++i) r[i] = b_local[i] - Av[i];
    double bnorm = global_norm(b_local, comm); if (bnorm < 1e-16) bnorm = 1.0;

    int total_iters = 0;
    std::vector<std::vector<double>> V(restart+1, std::vector<double>(n));
    std::vector<std::vector<double>> H(restart+1, std::vector<double>(restart));
    std::vector<double> cs(restart,0.0), sn(restart,0.0), e1(restart+1,0.0);

    double beta = global_norm(r, comm);
    if (beta / bnorm < tol) {
        if (out_iters) *out_iters = 0;
        if (out_final_res_norm) *out_final_res_norm = beta;
        return 0;
    }

    while (total_iters < max_iter) {
        // restart cycle
        e1.assign(restart+1,0.0);
        e1[0] = beta;
        for (int i = 0; i <= restart; ++i) for (int j = 0; j < restart; ++j) H[i][j] = 0.0;
        // v1 = r / beta
        for (int i = 0; i < n; ++i) V[0][i] = r[i] / beta;

        int m = 0;
        for (; m < restart && total_iters < max_iter; ++m) {
            // w = A * V[m]
            distributed_matvec(A, V[m], Av, comm);
            // apply left preconditioner to w if requested: (M^{-1} A V[m]) but for left precond we precondition r and p in algorithms.
            // here we keep classical GMRES without preconditioning for clarity; user can implement right-preconditioning externally.

            // Arnoldi: Modified Gram-Schmidt
            for (int j = 0; j <= m; ++j) {
                H[j][m] = global_dot(Av, V[j], comm);
                for (int i = 0; i < n; ++i) Av[i] -= H[j][m] * V[j][i];
            }
            H[m+1][m] = global_norm(Av, comm);
            if (H[m+1][m] < 1e-16) break;
            for (int i = 0; i < n; ++i) V[m+1][i] = Av[i] / H[m+1][m];

            // apply Givens rotations to new column
            for (int i = 0; i < m; ++i) {
                double temp = cs[i] * H[i][m] + sn[i] * H[i+1][m];
                H[i+1][m] = -sn[i] * H[i][m] + cs[i] * H[i+1][m];
                H[i][m] = temp;
            }
            // generate i-th rotation
            double rho = std::hypot(H[m][m], H[m+1][m]);
            if (rho < 1e-16) { cs[m] = 1.0; sn[m] = 0.0; }
            else { cs[m] = H[m][m] / rho; sn[m] = H[m+1][m] / rho; }
            // apply
            H[m][m] = cs[m] * H[m][m] + sn[m] * H[m+1][m];
            H[m+1][m] = 0.0;
            // update rhs
            double temp = cs[m] * e1[m] + sn[m] * e1[m+1];
            e1[m+1] = -sn[m] * e1[m] + cs[m] * e1[m+1];
            e1[m] = temp;

            double rel = std::abs(e1[m+1]) / bnorm;
            ++total_iters;
            if (rel < tol) {
                // solve upper triangular H(0:m,0:m) y = e1(0:m)
                std::vector<double> y(m+1,0.0);
                for (int i = m; i >= 0; --i) {
                    double s = e1[i];
                    for (int j = i+1; j <= m; ++j) s -= H[i][j] * y[j];
                    y[i] = s / H[i][i];
                }
                // x += V(:,0:m) * y
                for (int k = 0; k <= m; ++k) for (int i = 0; i < n; ++i) x_local[i] += V[k][i] * y[k];
                double final_res = std::abs(e1[m+1]);
                if (out_iters) *out_iters = total_iters;
                if (out_final_res_norm) *out_final_res_norm = final_res;
                return 0;
            }
        }
        // restart update: solve least squares for current H and update x
        int m_used = std::min(m, restart-1);
        if (m_used >= 0) {
            std::vector<double> y(m_used+1,0.0);
            for (int i = m_used; i >= 0; --i) {
                double s = e1[i];
                for (int j = i+1; j <= m_used; ++j) s -= H[i][j] * y[j];
                y[i] = s / H[i][i];
            }
            for (int k = 0; k <= m_used; ++k) for (int i = 0; i < n; ++i) x_local[i] += V[k][i] * y[k];
        }
        // recompute residual r = b - A*x
        distributed_matvec(A, x_local, r, comm);
        for (int i = 0; i < n; ++i) r[i] = b_local[i] - r[i];
        beta = global_norm(r, comm);
        if (beta / bnorm < tol) {
            if (out_iters) *out_iters = total_iters;
            if (out_final_res_norm) *out_final_res_norm = beta;
            return 0;
        }
    }
    if (out_iters) *out_iters = total_iters;
    if (out_final_res_norm) *out_final_res_norm = global_norm(r, comm);
    return 1;
}