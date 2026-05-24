/*
 * MPI Krylov Solver Project - Optimized Matrix Operations
 * 
 * Author: Xinye Chen (Enhanced with Communication Optimization)
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */

#include "matrix.hpp"
#include <algorithm>
#include <set>
#include <cmath>

// Initialize halo exchange pattern by analyzing matrix sparsity
void initialize_halo_exchange(CSRMatrix &A, MPI_Comm comm) {
    if (A.halo_initialized) return;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_start = A.row_offset;
    int local_end = A.row_offset + A.nrows;

    // Step 1: Find all non-local column indices needed
    std::set<int> halo_set;
    for (int i = 0; i < A.nrows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int col = A.col_idx[j];
            if (col < local_start || col >= local_end) {
                halo_set.insert(col);
            }
        }
    }

    A.halo_indices.assign(halo_set.begin(), halo_set.end());
    int halo_size = A.halo_indices.size();
    A.halo_data.resize(halo_size);

    // Step 2: Build halo_to_local mapping
    for (int i = 0; i < halo_size; ++i) {
        A.halo_to_local[A.halo_indices[i]] = i;
    }

    // Step 3: Determine which rank owns each halo index
    std::vector<int> rows_per_rank(size);
    int base_rows = A.ncols / size;
    int remainder = A.ncols % size;
    
    for (int r = 0; r < size; ++r) {
        rows_per_rank[r] = base_rows + (r < remainder ? 1 : 0);
    }

    std::vector<int> rank_offsets(size + 1, 0);
    for (int r = 0; r < size; ++r) {
        rank_offsets[r + 1] = rank_offsets[r] + rows_per_rank[r];
    }

    // Step 4: Build recv_map (which indices to receive from which rank)
    for (int global_idx : A.halo_indices) {
        for (int r = 0; r < size; ++r) {
            if (global_idx >= rank_offsets[r] && global_idx < rank_offsets[r + 1]) {
                A.recv_map[r].push_back(global_idx);
                break;
            }
        }
    }

    // Step 5: Exchange information to build send_map
    for (int r = 0; r < size; ++r) {
        int send_count = (r == rank) ? 0 : A.recv_map[r].size();
        int recv_count = 0;

        // Send how many indices we need from rank r
        MPI_Request req[2];
        if (r != rank) {
            MPI_Isend(&send_count, 1, MPI_INT, r, 0, comm, &req[0]);
        }
        
        // Receive how many indices rank r needs from us
        for (int src = 0; src < size; ++src) {
            if (src != rank) {
                MPI_Recv(&recv_count, 1, MPI_INT, src, 0, comm, MPI_STATUS_IGNORE);
                
                if (recv_count > 0) {
                    std::vector<int> requested_indices(recv_count);
                    MPI_Recv(requested_indices.data(), recv_count, MPI_INT, src, 1, comm, MPI_STATUS_IGNORE);
                    
                    // Convert global indices to local indices
                    for (int global_idx : requested_indices) {
                        int local_idx = global_idx - local_start;
                        if (local_idx >= 0 && local_idx < A.nrows) {
                            A.send_map[src].push_back(local_idx);
                        }
                    }
                }
            }
        }

        if (r != rank && send_count > 0) {
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            MPI_Send(A.recv_map[r].data(), send_count, MPI_INT, r, 1, comm);
        }
    }

    A.halo_initialized = true;
}

// Optimized distributed matvec with halo exchange
void distributed_matvec_optimized(const CSRMatrix &A, const std::vector<double> &x_local,
                                   std::vector<double> &y_local, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (!A.halo_initialized) {
        const_cast<CSRMatrix&>(A).halo_initialized = false;
        initialize_halo_exchange(const_cast<CSRMatrix&>(A), comm);
    }

    y_local.resize(A.nrows);
    std::fill(y_local.begin(), y_local.end(), 0.0);

    // Step 1: Non-blocking send/recv for halo exchange
    std::vector<MPI_Request> requests;
    std::vector<std::vector<double>> send_buffers;

    for (auto &entry : A.send_map) {
        int dest_rank = entry.first;
        const auto &local_indices = entry.second;
        
        send_buffers.emplace_back(local_indices.size());
        for (size_t i = 0; i < local_indices.size(); ++i) {
            send_buffers.back()[i] = x_local[local_indices[i]];
        }

        MPI_Request req;
        MPI_Isend(send_buffers.back().data(), local_indices.size(), MPI_DOUBLE, 
                  dest_rank, 0, comm, &req);
        requests.push_back(req);
    }

    for (auto &entry : A.recv_map) {
        int src_rank = entry.first;
        const auto &global_indices = entry.second;
        
        std::vector<double> recv_buffer(global_indices.size());
        MPI_Request req;
        MPI_Irecv(recv_buffer.data(), global_indices.size(), MPI_DOUBLE,
                  src_rank, 0, comm, &req);
        requests.push_back(req);

        // Store buffer for later use (need to persist until Waitall)
        send_buffers.push_back(std::move(recv_buffer));
    }

    // Step 2: Compute local part while communication happens
    int local_start = A.row_offset;
    int local_end = A.row_offset + A.nrows;

    for (int i = 0; i < A.nrows; ++i) {
        double sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int col = A.col_idx[j];
            if (col >= local_start && col < local_end) {
                sum += A.values[j] * x_local[col - local_start];
            }
        }
        y_local[i] = sum;
    }

    // Step 3: Wait for all communications to complete
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    // Step 4: Unpack halo data
    size_t recv_buf_idx = A.send_map.size();
    for (auto &entry : A.recv_map) {
        const auto &global_indices = entry.second;
        const auto &recv_buffer = send_buffers[recv_buf_idx++];
        
        for (size_t i = 0; i < global_indices.size(); ++i) {
            int halo_pos = A.halo_to_local.at(global_indices[i]);
            const_cast<CSRMatrix&>(A).halo_data[halo_pos] = recv_buffer[i];
        }
    }

    // Step 5: Add non-local contributions
    for (int i = 0; i < A.nrows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int col = A.col_idx[j];
            if (col < local_start || col >= local_end) {
                int halo_pos = A.halo_to_local.at(col);
                y_local[i] += A.values[j] * A.halo_data[halo_pos];
            }
        }
    }
}

// Original simple version (for compatibility and small problems)
void distributed_matvec(const CSRMatrix &A, const std::vector<double> &x_local,
                        std::vector<double> &y_local, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int N = A.ncols;
    std::vector<double> x_full(N);
    std::vector<int> recvcounts(size), displs(size);

    int base = N / size;
    int rem = N % size;
    for (int r = 0; r < size; ++r) {
        recvcounts[r] = base + (r < rem ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + recvcounts[r - 1];
    }

    MPI_Allgatherv(x_local.data(), A.nrows, MPI_DOUBLE,
                   x_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, comm);

    y_local.resize(A.nrows);
    for (int i = 0; i < A.nrows; ++i) {
        double sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * x_full[A.col_idx[j]];
        }
        y_local[i] = sum;
    }
}

// Global dot product
double global_dot(const std::vector<double> &a_local, const std::vector<double> &b_local, MPI_Comm comm) {
    double local_sum = 0.0;
    int n = a_local.size();
    for (int i = 0; i < n; ++i) {
        local_sum += a_local[i] * b_local[i];
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum;
}

// Global norm
double global_norm(const std::vector<double> &a_local, MPI_Comm comm) {
    double local_sum = 0.0;
    for (double val : a_local) {
        local_sum += val * val;
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(global_sum);
}

// Multi-dot for reduced communication (used in CA methods)
void global_multi_dot(const std::vector<std::vector<double>> &vecs_a,
                      const std::vector<std::vector<double>> &vecs_b,
                      std::vector<double> &results, MPI_Comm comm) {
    int num_dots = vecs_a.size();
    results.resize(num_dots);
    
    std::vector<double> local_dots(num_dots, 0.0);
    
    for (int k = 0; k < num_dots; ++k) {
        int n = vecs_a[k].size();
        for (int i = 0; i < n; ++i) {
            local_dots[k] += vecs_a[k][i] * vecs_b[k][i];
        }
    }
    
    MPI_Allreduce(local_dots.data(), results.data(), num_dots, MPI_DOUBLE, MPI_SUM, comm);
}