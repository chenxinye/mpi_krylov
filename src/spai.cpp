/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 

#include "spai.hpp"

SPAIPrecond::SPAIPrecond(const CSRMatrix &/*A*/) {}
void SPAIPrecond::apply(const std::vector<double>& r_local, std::vector<double>& z_local) { z_local = r_local; }
