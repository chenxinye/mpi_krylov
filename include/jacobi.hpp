#ifndef JACOBI_HPP
#define JACOBI_HPP

#include <vector>
#include "preconditioner.hpp"
#include "matrix.hpp"

struct JacobiPrecond : public Preconditioner {
    std::vector<double> invdiag_local;

    JacobiPrecond(const CSRMatrix& A);

    void apply(const std::vector<double>& r_local, std::vector<double>& z_local) override;
};

#endif