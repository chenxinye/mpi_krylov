#include <stdexcept>

JacobiPrecond::JacobiPrecond(const CSRMatrix &A) {
    invdiag_local.assign(A.local_n(), 0.0);
    for (int i = 0; i < A.local_n(); ++i) {
        bool found = false;
        for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; ++jj) {
            if (A.col_idx[jj] == A.row_offset + i) {
                double d = A.values[jj];
                if (d == 0.0) throw std::runtime_error("Zero diagonal in Jacobi preconditioner");
                invdiag_local[i] = 1.0 / d;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("No diagonal entry found for Jacobi preconditioner (local row)");
    }
}

void JacobiPrecond::apply(const std::vector<double>& r_local, std::vector<double>& z_local) {
    z_local.resize(r_local.size());
    for (size_t i = 0; i < r_local.size(); ++i) z_local[i] = invdiag_local[i] * r_local[i];
}
