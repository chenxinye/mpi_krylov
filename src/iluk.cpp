/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 

#include "iluk.hpp"

ILUkPrecond::ILUkPrecond(const CSRMatrix &/*A*/, int k) : klevel(k) {}
void ILUkPrecond::apply(const std::vector<double>& r_local, std::vector<double>& z_local) { z_local = r_local; }
