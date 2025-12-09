#ifndef BLOCK_JACOBI_PRECOND_HPP
#define BLOCK_JACOBI_PRECOND_HPP
#include <algorithm>
#include <vector>
#include <cmath>
#include "matrix.hpp"
#include "preconditioner.hpp"
#include "preconditioner.hpp"

class BlockJacobiPrecond : public Preconditioner {
    int block_size;
    std::vector<std::vector<double>> inv_blocks; // dense inverse per block (local)
    int nlocal;
public:
    BlockJacobiPrecond(const CSRMatrix &A, int block_size_);
    void apply(const std::vector<double>& r_local, std::vector<double>& z_local) override;
    long long nnz_after() const override { // approximate dense blocks count
        long long s = 0;
        for (auto &b : inv_blocks) s += (long long)b.size();
        return s;
    }
};

#endif
