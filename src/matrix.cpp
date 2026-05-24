/*
 * MPI Krylov Solver Project - Optimized Matrix Operations
 * FIXED: Deadlock-free halo exchange initialization
 */

#include "matrix.hpp"
#include <algorithm>
#include <set>
#include <cmath>
#include <iostream>

// Initialize halo exchange pattern - COMPLETELY REWRITTEN
void initialize_halo_exchange(CSRMatrix &A, MPI_Comm comm) {
    if (A.halo_initialized) return;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_start = A.row_offset;
    int local_end = A.row_offset + A.nrows;

    // Step 1: Find non-local columns
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

    for (int i = 0; i < halo_size; ++i) {
        A.halo_to_local[A.halo_indices[i]] = i;
    }

    // Step 2: Determine row distribution
    std::vector<int> rank_offsets(size + 1, 0);
    int base_rows = A.ncols / size;
    int remainder = A.ncols % size;
    
    for (int r = 0; r < size; ++r) {
        int rows_in_rank = base_rows + (r < remainder ? 1 : 0);
        rank_offsets[r + 1] = rank_offsets[r] + rows_in_rank;
    }

    // Step 3: Build recv_map
    for (int global_idx : A.halo_indices) {
        for (int r = 0; r < size; ++r) {
            if (global_idx >= rank_offsets[r] && global_idx < rank_offsets[r + 1]) {
                A.recv_map[r].push_back(global_idx);
                break;
            }
        }
    }

    // Step 4: Exchange to build send_map - FIXED DEADLOCK
    // Use MPI_Alltoall pattern to avoid deadlock
    
    // First, exchange counts
    std::vector<int> send_counts(size, 0);
    std::vector<int> recv_counts(size, 0);
    
    for (int r = 0; r < size; ++r) {
        send_counts[r] = (r == rank) ? 0 : A.recv_map[r].size();
    }
    
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, comm);
    
    // Prepare send buffers
    std::vector<int> send_displs(size + 1, 0);
    std::vector<int> recv_displs(size + 1, 0);
    
    for (int r = 0; r < size; ++r) {
        send_displs[r + 1] = send_displs[r] + send_counts[r];
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
    }
    
    int total_send = send_displs[size];
    int total_recv = recv_displs[size];
    
    std::vector<int> send_buffer(total_send);
    std::vector<int> recv_buffer(total_recv);
    
    // Pack send buffer
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        if (r != rank) {
            for (int idx : A.recv_map[r]) {
                send_buffer[offset++] = idx;
            }
        }
    }
    
    // Exchange all data at once
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_INT, comm);
    
    // Unpack and build send_map
    offset = 0;
    for (int r = 0; r < size; ++r) {
        if (r != rank && recv_counts[r] > 0) {
            for (int i = 0; i < recv_counts[r]; ++i) {
                int global_idx = recv_buffer[offset++];
                int local_idx = global_idx - local_start;
                if (local_idx >= 0 && local_idx < A.nrows) {
                    A.send_map[r].push_back(local_idx);
                }
            }
        }
    }

    A.halo_initialized = true;
}

// Optimized distributed matvec
void distributed_matvec_optimized(const CSRMatrix &A, const std::vector<double> &x_local,
                                   std::vector<double> &y_local, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (!A.halo_initialized) {
        distributed_matvec(A, x_local, y_local, comm);
        return;
    }

    y_local.resize(A.nrows);
    std::fill(y_local.begin(), y_local.end(), 0.0);

    // Use MPI_Alltoallv for symmetric exchange
    std::vector<int> send_counts(size, 0);
    std::vector<int> recv_counts(size, 0);
    std::vector<int> send_displs(size + 1, 0);
    std::vector<int> recv_displs(size + 1, 0);

    for (auto &entry : A.send_map) {
        send_counts[entry.first] = entry.second.size();
    }
    
    for (auto &entry : A.recv_map) {
        recv_counts[entry.first] = entry.second.size();
    }
    
    for (int r = 0; r < size; ++r) {
        send_displs[r + 1] = send_displs[r] + send_counts[r];
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
    }
    
    int total_send = send_displs[size];
    int total_recv = recv_displs[size];
    
    std::vector<double> send_buffer(total_send);
    std::vector<double> recv_buffer(total_recv);
    
    // Pack send data
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        if (A.send_map.count(r)) {
            for (int local_idx : A.send_map.at(r)) {
                send_buffer[offset++] = x_local[local_idx];
            }
        }
    }

    // Compute local part during communication setup
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

    // Exchange halo data
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE, comm);

    // Unpack and add non-local contributions
    offset = 0;
    for (int r = 0; r < size; ++r) {
        if (A.recv_map.count(r)) {
            const auto &global_indices = A.recv_map.at(r);
            for (size_t i = 0; i < global_indices.size(); ++i) {
                int halo_pos = A.halo_to_local.at(global_indices[i]);
                const_cast<CSRMatrix&>(A).halo_data[halo_pos] = recv_buffer[offset++];
            }
        }
    }

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

// Original simple version
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

double global_norm(const std::vector<double> &a_local, MPI_Comm comm) {
    double local_sum = 0.0;
    for (double val : a_local) {
        local_sum += val * val;
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(global_sum);
}

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