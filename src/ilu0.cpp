
#include "ilu0.hpp"

ILU0Precond::ILU0Precond(const CSRMatrix &A) {
    nlocal = A.local_n();
    L.assign(nlocal*nlocal, 0.0);
    U.assign(nlocal*nlocal, 0.0);
    valid = false;
    if (nlocal == 0) return;
    // form dense local A_local using only local columns (this is a simplification)
    std::vector<double> Aloc(nlocal*nlocal, 0.0);
    for (int i = 0; i < nlocal; ++i) {
        for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; ++jj) {
            int col = A.col_idx[jj] - A.row_offset; // local index
            if (col >= 0 && col < nlocal) Aloc[i*nlocal + col] = A.values[jj];
        }
    }
    // Copy Aloc to U initially
    U = Aloc;
    // perform Doolittle ILU(0) with same sparsity as Aloc (dense here)
    for (int k = 0; k < nlocal; ++k) {
        double diag = U[k*nlocal + k];
        if (std::abs(diag) < 1e-14) { valid = false; return; }
        for (int i = k+1; i < nlocal; ++i) {
            double lik = U[i*nlocal + k] / diag;
            L[i*nlocal + k] = lik;
            for (int j = k; j < nlocal; ++j) {
                U[i*nlocal + j] -= lik * U[k*nlocal + j];
            }
        }
    }
    // set unit diagonal of L
    for (int i = 0; i < nlocal; ++i) L[i*nlocal + i] = 1.0;
    valid = true;
}

void ILU0Precond::apply(const std::vector<double>& r_local, std::vector<double>& z_local) {
    if (!valid) { z_local = r_local; return; }
    // solve L y = r (forward)
    std::vector<double> y(nlocal, 0.0);
    for (int i = 0; i < nlocal; ++i) {
        double s = r_local[i];
        for (int j = 0; j < i; ++j) s -= L[i*nlocal + j] * y[j];
        y[i] = s / L[i*nlocal + i];
    }
    // solve U z = y (backward)
    z_local.assign(nlocal, 0.0);
    for (int i = nlocal-1; i >= 0; --i) {
        double s = y[i];
        for (int j = i+1; j < nlocal; ++j) s -= U[i*nlocal + j] * z_local[j];
        z_local[i] = s / U[i*nlocal + i];
    }
}
