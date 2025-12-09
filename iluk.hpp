#ifndef ILUK_PRECOND_HPP
#define ILUK_PRECOND_HPP

#include "preconditioner.hpp"

class ILUkPrecond : public Preconditioner {
    int klevel;
public:
    ILUkPrecond(const CSRMatrix &A, int k);
    void apply(const std::vector<double>& r_local, std::vector<double>& z_local) override;
};

#endif