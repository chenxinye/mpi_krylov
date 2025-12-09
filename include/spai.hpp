#ifndef SPAI_PRECOND_HPP
#define SPAI_PRECOND_HPP

#include <vector>
#include <cmath>
#include "matrix.hpp"
#include "preconditioner.hpp"

class SPAIPrecond : public Preconditioner {
public:
    SPAIPrecond(const CSRMatrix &A);
    void apply(const std::vector<double>& r_local, std::vector<double>& z_local) override;
};

#endif
