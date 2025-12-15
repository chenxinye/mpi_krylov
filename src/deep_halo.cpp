/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 
// implementation (naive dense block invert for demonstration)


#include "matrix.hpp"
#include <vector>
#include <mpi.h>
#include "ca_matrix.hpp"
#include <algorithm>

void exchange_deep_halo(const std::vector<double>& x_local, 
                        std::vector<double>& x_extended, 
                        DeepHaloContext& ctx, 
                        MPI_Comm comm) {
    
    std::vector<std::vector<double>> send_buffers(ctx.send_plans.size());
    std::vector<MPI_Request> reqs;

    for (size_t i = 0; i < ctx.send_plans.size(); ++i) {
        const auto& plan = ctx.send_plans[i];
        send_buffers[i].resize(plan.src_global_indices.size());
        
        for (size_t j = 0; j < plan.src_global_indices.size(); ++j) {
            int local_idx = plan.src_global_indices[j]; 
            send_buffers[i][j] = x_local[local_idx];
        }

        reqs.emplace_back();
        MPI_Isend(send_buffers[i].data(), send_buffers[i].size(), MPI_DOUBLE, 
                  plan.rank, 0, comm, &reqs.back());
    }

    std::copy(x_local.begin(), x_local.end(), x_extended.begin());

    for (const auto& plan : ctx.receive_plans) {
        std::vector<double> recv_buf(plan.dest_local_indices.size());
        MPI_Status status;
        
        MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_DOUBLE, 
                 plan.rank, 0, comm, &status);

        for (size_t j = 0; j < recv_buf.size(); ++j) {
            int target_idx = plan.dest_local_indices[j];
            x_extended[target_idx] = recv_buf[j];
        }
    }

    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}