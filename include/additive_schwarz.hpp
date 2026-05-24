/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 

#ifndef ADDITIVE_SCHWARZ_HPP
#define ADDITIVE_SCHWARZ_HPP

#include "preconditioner.hpp"
#include "matrix.hpp"
#include <vector>

/**
 * Additive Schwarz Preconditioner
 * Each MPI rank solves a local subdomain problem independently
 */
class AdditiveSchwarzPrecond : public Preconditioner {
private:
    int local_size;
    
    // Local subdomain matrix (dense for simplicity)
    std::vector<double> local_A_dense;
    std::vector<double> local_A_inv;  // LU factorization storage
    std::vector<int> pivot;
    
    bool factorized;
    
    // LU factorization of local subdomain
    void factorize_local_subdomain();

public:
    /**
     * Constructor
     * @param A: Reference to distributed matrix
     * @param overlap_size: Number of overlapping rows (currently unused, reserved for future)
     * @param mpi_comm: MPI communicator (currently unused, reserved for future)
     */
    AdditiveSchwarzPrecond(const CSRMatrix& A, int overlap_size, MPI_Comm mpi_comm);
    
    void apply(const std::vector<double>& r_local, std::vector<double>& z_local) override;
    
    long long nnz_after() const override { 
        return factorized ? (long long)(local_size * local_size) : 0; 
    }
};

#endif