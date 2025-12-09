#ifndef GMRES_SOLVER_HPP
#define GMRES_SOLVER_HPP

#include <vector>
#include <cmath>
#include <mpi.h>            
#include "matrix.hpp"        
#include "preconditioner.hpp" 


int gmres_solve(const CSRMatrix& A, const std::vector<double>& b_local, std::vector<double>& x_local,
                int restart, int max_iter, double tol, MPI_Comm comm, Preconditioner* M=nullptr,
                int *out_iters=nullptr, double *out_final_res_norm=nullptr);

#endif

