// matrix.hpp (追加或修改)
#ifndef MATRIX_DEEP_HALO_HPP
#define MATRIX_DEEP_HALO_HPP

#include <vector>
#include <map>
#include <mpi.h>
#include "matrix.hpp"

struct DeepHaloContext {
    int s_depth;        
    int total_ghosts;    
    
    std::map<int, int> global_to_local_map;

    struct CommPlan {
        int rank;
        std::vector<int> src_global_indices; 
        std::vector<int> dest_local_indices;
    };
    std::vector<CommPlan> receive_plans;
    std::vector<CommPlan> send_plans;

    std::vector<CSRMatrix> ghost_layers_matrices; 

    std::vector<int> layer_counts; 

    DeepHaloContext(int s) : s_depth(s), total_ghosts(0) {}
};

// 增加函数声明
void setup_deep_halo(const CSRMatrix& local_A, int s, DeepHaloContext& ctx, MPI_Comm comm);
void exchange_deep_halo(const std::vector<double>& x_local, std::vector<double>& x_extended, DeepHaloContext& ctx, MPI_Comm comm);

#endif