/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 
// implementation (naive dense block invert for demonstration)


#include "ca_matrix.hpp"
#include <algorithm>
#include <vector>

void exchange_deep_halo(const std::vector<double>& x_local, 
                        std::vector<double>& x_extended, 
                        DeepHaloContext& ctx, 
                        MPI_Comm comm) {
    
    std::copy(x_local.begin(), x_local.end(), x_extended.begin());

    std::vector<MPI_Request> reqs;
    std::vector<std::vector<double>> send_buffers(ctx.send_plans.size());
    std::vector<std::vector<double>> recv_buffers(ctx.receive_plans.size());

    for (size_t i = 0; i < ctx.send_plans.size(); ++i) {
        const auto& plan = ctx.send_plans[i];
        send_buffers[i].resize(plan.src_global_indices.size());
        
        for (size_t j = 0; j < plan.src_global_indices.size(); ++j) {
            int idx = plan.src_global_indices[j];
            send_buffers[i][j] = x_local[idx];
        }

        MPI_Request req;
        MPI_Isend(send_buffers[i].data(), send_buffers[i].size(), MPI_DOUBLE, 
                  plan.rank, 0, comm, &req);
        reqs.push_back(req);
    }

    for (size_t i = 0; i < ctx.receive_plans.size(); ++i) {
        const auto& plan = ctx.receive_plans[i];
        recv_buffers[i].resize(plan.dest_local_indices.size());

        MPI_Request req;
        MPI_Irecv(recv_buffers[i].data(), recv_buffers[i].size(), MPI_DOUBLE, 
                  plan.rank, 0, comm, &req);
        reqs.push_back(req);
    }

    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    for (size_t i = 0; i < ctx.receive_plans.size(); ++i) {
        const auto& plan = ctx.receive_plans[i];
        const auto& buf = recv_buffers[i];
        
        for (size_t j = 0; j < buf.size(); ++j) {
            int target_idx = plan.dest_local_indices[j]; 
            x_extended[target_idx] = buf[j];
        }
    }
}