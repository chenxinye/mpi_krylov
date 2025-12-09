#ifndef ILU0_PRECOND_HPP
#define ILU0_PRECOND_HPP

#include "preconditioner.hpp"

class ILU0Precond : public Preconditioner {
    // store L and U as dense local matrices (nlocal x nlocal) for demo simplicity
    int nlocal;
    std::vector<double> L; // unit lower: 1 on diagonal implicitly
    std::vector<double> U; // upper including diagonal
    bool valid;
public:
    ILU0Precond(const CSRMatrix &A);
    void apply(const std::vector<double>& r_local, std::vector<double>& z_local) override;
    long long nnz_after() const override { return valid ? (long long)(nlocal*nlocal) : -1; }
};

#endif