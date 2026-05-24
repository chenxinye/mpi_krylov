#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <mpi.h>
#include <map>

struct CSRMatrix {
    int nrows = 0;              // local number of rows
    int ncols = 0;              // global number of columns
    int row_offset = 0;         // global index of first local row
    std::vector<int> row_ptr;   // size nrows+1
    std::vector<int> col_idx;   // global column indices
    std::vector<double> values; // nonzero values

    // Halo exchange data structures
    std::vector<int> halo_indices;           // global indices needed from other ranks
    std::map<int, int> halo_to_local;        // map global index -> local halo position
    std::vector<double> halo_data;           // received halo values
    std::map<int, std::vector<int>> send_map; // rank -> list of local indices to send
    std::map<int, std::vector<int>> recv_map; // rank -> list of halo indices to receive
    bool halo_initialized = false;

    int local_n() const { return nrows; }
    int global_n() const { return ncols; }
    long long nnz() const { return (long long)values.size(); }
};

// Initialize halo exchange pattern
void initialize_halo_exchange(CSRMatrix &A, MPI_Comm comm);

// Optimized distributed matvec with halo exchange
void distributed_matvec_optimized(const CSRMatrix &A, const std::vector<double> &x_local, 
                                   std::vector<double> &y_local, MPI_Comm comm);

// Original simple version (kept for compatibility)
void distributed_matvec(const CSRMatrix &A, const std::vector<double> &x_local, 
                        std::vector<double> &y_local, MPI_Comm comm);

// Non-blocking version for computation-communication overlap
void distributed_matvec_nonblocking(const CSRMatrix &A, const std::vector<double> &x_local,
                                     std::vector<double> &y_local, MPI_Comm comm,
                                     std::vector<MPI_Request> &requests);

// Global dot product and norm helpers
double global_dot(const std::vector<double>& a_local, const std::vector<double>& b_local, MPI_Comm comm);
double global_norm(const std::vector<double>& a_local, MPI_Comm comm);

// Multi-dot for reduced communication (CA methods)
void global_multi_dot(const std::vector<std::vector<double>>& vecs_a,
                      const std::vector<std::vector<double>>& vecs_b,
                      std::vector<double>& results, MPI_Comm comm);

#endif