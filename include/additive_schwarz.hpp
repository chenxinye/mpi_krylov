#ifndef ADDITIVE_SCHWARZ_HPP
#define ADDITIVE_SCHWARZ_HPP

#include "preconditioner.hpp"
#include "matrix.hpp"
#include <vector>

/**
 * Additive Schwarz Preconditioner
 * Each MPI rank solves a local subdomain problem independently
 * with optional overlap for better convergence
 */
class AdditiveSchwarzPrecond : public Preconditioner {
private:
    int local_size;
    int overlap;  // overlap size with neighboring domains
    
    // Local subdomain matrix (dense for simplicity, could be sparse)
    std::vector<double> local_A_dense;
    std::vector<double> local_A_inv;  // LU factorization storage
    std::vector<int> pivot;
    
    bool factorized;
    MPI_Comm comm;
    
    // LU factorization of local subdomain
    void factorize_local_subdomain(const CSRMatrix &A);
    
    // Solve local system using LU factors
    void solve_local(const std::vector<double> &rhs, std::vector<double> &solution);

public:
    /**
     * Constructor
     * @param A: Distributed CSR matrix
     * @param overlap_size: Number of overlapping rows with neighbors (0 = no overlap)
     * @param mpi_comm: MPI communicator
     */
    AdditiveSchwarzPrecond(const CSRMatrix &A, int overlap_size, MPI_Comm mpi_comm);
    
    void apply(const std::vector<double> &r_local, std::vector<double> &z_local) override;
    
    long long nnz_after() const override { 
        return factorized ? (long long)(local_size * local_size) : 0; 
    }
};

#endif