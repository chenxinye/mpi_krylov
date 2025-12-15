/*
 * MPI Krylov Solver Project
 * 
 * Author: Xinye Chen
 * Affiliation: Postdoctoral Researcher, Sorbonne University, LIP6, CNRS
 */ 
// implementation (naive dense block invert for demonstration)


#ifndef CA_STRUCTS_HPP
#define CA_STRUCTS_HPP

#include <vector>
#include <mpi.h>
#include "matrix.hpp" 

// Tall-Skinny Matrix)
// V = [v_0, v_1, ..., v_s]
struct CABlockVector {
    int num_rows;    
    int num_cols;    
    std::vector<double> data; 

    CABlockVector(int n, int s) : num_rows(n), num_cols(s), data(n * s, 0.0) {}

    double& at(int i, int j) {
        return data[j * num_rows + i];
    }

    const double& at(int i, int j) const {
        return data[j * num_rows + i];
    }
    
    double* col_ptr(int j) {
        return &data[j * num_rows];
    }
    
    const double* col_ptr(int j) const {
        return &data[j * num_rows];
    }
};

#endif