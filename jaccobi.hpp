#ifndef JACOBI_PRECOND_HPP
#define JACOBI_PRECOND_HPP

#include "preconditioner.hpp"

class JacobiPrecond : public Preconditioner {
    std::vector<double> invdiag_local; // inverse of diagonal for local rows
public:
    JacobiPrecond(const CSRMatrix &A);
    void apply(const std::vector<double>& r_local, std::vector<double>& z_local) override;
    long long nnz_after() const override { return (long long)invdiag_local.size(); }
};

#endif