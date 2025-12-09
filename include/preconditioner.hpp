#ifndef PRECONDITIONER_HPP
#define PRECONDITIONER_HPP

#include <vector>
#include "matrix.hpp"

struct Preconditioner {
    virtual void apply(const std::vector<double>& r_local, std::vector<double>& z_local) = 0;
    virtual ~Preconditioner() = default;
    virtual long long nnz_after() const { return 0; }
};


#endif
