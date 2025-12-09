#ifndef PRECONDITIONER_HPP
#define PRECONDITIONER_HPP

#include <vector>
#include "matrix.hpp"

struct Preconditioner {
    virtual void apply(const std::vector<double>& r_local, std::vector<double>& z_local) = 0;
    virtual ~Preconditioner() = default;
};

struct JacobiPrecond : public Preconditioner {
    std::vector<double> inv_diag;

    JacobiPrecond(const CSRMatrix& A) {
        inv_diag.resize(A.nrows);
        for (int i = 0; i < A.nrows; ++i) {
            inv_diag[i] = 0.0;
            for (int j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j) {
                if (A.col_idx[j] == i) inv_diag[i] = 1.0 / A.values[j];
            }
        }
    }

    void apply(const std::vector<double>& r_local, std::vector<double>& z_local) override {
        int n = r_local.size();
        z_local.resize(n);
        for (int i = 0; i < n; ++i) z_local[i] = inv_diag[i] * r_local[i];
    }
};

#endif
