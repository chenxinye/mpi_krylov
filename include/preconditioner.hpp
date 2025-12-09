#ifndef PRECONDITIONER_HPP
#define PRECONDITIONER_HPP
#include <vector>
#include "matrix.hpp"

class Preconditioner {
public:
    // supply A in constructor for setup; apply: z = M^{-1} r (approx)
    virtual void apply(const std::vector<double>& r_local, std::vector<double>& z_local) = 0;
    virtual long long nnz_after() const { return -1; } // optionally report nnz of factor/approx
    virtual ~Preconditioner() = default;
};

#endif