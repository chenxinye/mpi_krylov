/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 
// implementation (naive dense block invert for demonstration)

#include "block_jacobi.hpp"

BlockJacobiPrecond::BlockJacobiPrecond(const CSRMatrix &A, int block_size_)
    : block_size(block_size_), nlocal(A.local_n()) {
    int nblocks = (nlocal + block_size - 1) / block_size;
    inv_blocks.resize(nblocks);
    // Construct dense block and invert naively (Gauss-Jordan) per block using local rows
    for (int b = 0; b < nblocks; ++b) {
        int istart = b * block_size;
        int iend = std::min(istart + block_size, nlocal);
        int sz = iend - istart;
        if (sz <= 0) continue;
        std::vector<double> M(sz * sz, 0.0);
        // fill M from A (local part only)
        for (int i = istart; i < iend; ++i) {
            for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; ++jj) {
                int col = A.col_idx[jj] - A.row_offset; // local index or remote
                if (col >= istart && col < iend) {
                    M[(i - istart) * sz + (col - istart)] = A.values[jj];
                }
            }
        }
        // invert M in-place to get inv(M)
        // build augmented matrix [M | I]
        std::vector<double> aug(sz * (2*sz), 0.0);
        for (int i = 0; i < sz; ++i) {
            for (int j = 0; j < sz; ++j) aug[i*(2*sz)+j] = M[i*sz+j];
            aug[i*(2*sz) + sz + i] = 1.0;
        }
        // Gauss-Jordan
        for (int col = 0; col < sz; ++col) {
            // pivot
            double piv = aug[col*(2*sz)+col];
            if (std::abs(piv) < 1e-14) {
                // singular or nearly singular block: use identity
                inv_blocks[b].assign(sz*sz, 0.0);
                for (int i = 0; i < sz; ++i) inv_blocks[b][i*sz + i] = 1.0;
                goto next_block;
            }
            double invp = 1.0 / piv;
            for (int j = 0; j < 2*sz; ++j) aug[col*(2*sz)+j] *= invp;
            for (int i = 0; i < sz; ++i) if (i != col) {
                double fac = aug[i*(2*sz)+col];
                if (fac == 0.0) continue;
                for (int j = 0; j < 2*sz; ++j) aug[i*(2*sz)+j] -= fac * aug[col*(2*sz)+j];
            }
        }
        // extract inverse
        inv_blocks[b].assign(sz*sz, 0.0);
        for (int i = 0; i < sz; ++i)
            for (int j = 0; j < sz; ++j)
                inv_blocks[b][i*sz + j] = aug[i*(2*sz) + sz + j];
    next_block: ;
    }
}

void BlockJacobiPrecond::apply(const std::vector<double>& r_local, std::vector<double>& z_local) {
    z_local = r_local;
    int nblocks = (nlocal + block_size - 1) / block_size;
    for (int b = 0; b < nblocks; ++b) {
        int istart = b * block_size;
        int iend = std::min(istart + block_size, nlocal);
        int sz = iend - istart;
        if (sz <= 0) continue;
        std::vector<double> tmp(sz, 0.0);
        for (int i = 0; i < sz; ++i)
            for (int j = 0; j < sz; ++j)
                tmp[i] += inv_blocks[b][i*sz+j] * r_local[istart + j];
        for (int i = 0; i < sz; ++i) z_local[istart + i] = tmp[i];
    }
}