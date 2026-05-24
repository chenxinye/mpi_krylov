/*
 * MPI Krylov Solver Project
 * 
 * Polynomial Preconditioner - Communication-free preconditioning
 */

#include "polynomial_precond.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

PolynomialPrecond::PolynomialPrecond(const CSRMatrix& A, PolyType poly_type, int poly_degree, MPI_Comm mpi_comm)
    : A_ptr(&A), comm(mpi_comm), type(poly_type), degree(poly_degree) {
    
    if (degree < 1) degree = 1;
    if (degree > 15) degree = 15;  // Practical limit
    
    int n = A.local_n();
    diag_inv.resize(n);
    
    // Extract diagonal for scaling
    for (int i = 0; i < A.nrows; ++i) {
        double diag_val = 0.0;
        int global_i = A.row_offset + i;
        
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_idx[j] == global_i) {
                diag_val = A.values[j];
                break;
            }
        }
        
        diag_inv[i] = (std::abs(diag_val) > 1e-14) ? (1.0 / diag_val) : 1.0;
    }
    
    if (type == CHEBYSHEV) {
        estimate_spectrum();
        compute_chebyshev_coeffs();
    } else {
        // Neumann series: simple coefficients
        coeffs.resize(degree + 1, 1.0);
    }
}

void PolynomialPrecond::estimate_spectrum() {
    int n = A_ptr->local_n();
    std::vector<double> v(n, 1.0);
    std::vector<double> Av(n);
    
    // Normalize initial vector
    double norm = global_norm(v, comm);
    for (int i = 0; i < n; ++i) v[i] /= norm;
    
    // Power iteration for largest eigenvalue
    double lambda_max_est = 0.0;
    for (int iter = 0; iter < 20; ++iter) {
        distributed_matvec(*A_ptr, v, Av, comm);
        lambda_max_est = global_dot(v, Av, comm);
        
        double Av_norm = global_norm(Av, comm);
        if (Av_norm > 1e-14) {
            for (int i = 0; i < n; ++i) v[i] = Av[i] / Av_norm;
        }
    }
    lambda_max = std::abs(lambda_max_est);
    
    // Estimate smallest eigenvalue (inverse iteration would be more accurate)
    // For simplicity, use Gershgorin estimate
    double min_diag = 1e10;
    for (int i = 0; i < A_ptr->nrows; ++i) {
        int global_i = A_ptr->row_offset + i;
        double diag_val = 0.0;
        double off_diag_sum = 0.0;
        
        for (int j = A_ptr->row_ptr[i]; j < A_ptr->row_ptr[i + 1]; ++j) {
            if (A_ptr->col_idx[j] == global_i) {
                diag_val = A_ptr->values[j];
            } else {
                off_diag_sum += std::abs(A_ptr->values[j]);
            }
        }
        
        double lower_bound = std::abs(diag_val) - off_diag_sum;
        min_diag = std::min(min_diag, lower_bound);
    }
    
    double global_min_diag = 0.0;
    MPI_Allreduce(&min_diag, &global_min_diag, 1, MPI_DOUBLE, MPI_MIN, comm);
    lambda_min = std::max(global_min_diag, lambda_max * 0.01);  // Safety bound
    
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::cout << "[PolyPrecond] Estimated spectrum: [" << lambda_min 
                  << ", " << lambda_max << "]\n";
    }
}

void PolynomialPrecond::compute_chebyshev_coeffs() {
    // Chebyshev polynomial on [lambda_min, lambda_max]
    double center = (lambda_max + lambda_min) / 2.0;
    double radius = (lambda_max - lambda_min) / 2.0;
    
    coeffs.resize(degree + 1);
    
    // Simplified Chebyshev coefficients (first-kind)
    // For production: use recurrence relation for accurate coefficients
    for (int k = 0; k <= degree; ++k) {
        coeffs[k] = std::pow(-1.0, k) / (center + radius * std::cos(M_PI * k / degree));
    }
    
    // Normalize
    double sum = 0.0;
    for (double c : coeffs) sum += c;
    if (std::abs(sum) > 1e-14) {
        for (double& c : coeffs) c /= sum;
    }
}

void PolynomialPrecond::apply(const std::vector<double>& r_local, std::vector<double>& z_local) {
    int n = A_ptr->local_n();
    z_local.assign(n, 0.0);
    
    if (type == NEUMANN) {
        // Neumann series: z = (I + D^{-1}A + (D^{-1}A)^2 + ...) D^{-1} r
        // Compute D^{-1} r first
        std::vector<double> scaled_r(n);
        for (int i = 0; i < n; ++i) {
            scaled_r[i] = diag_inv[i] * r_local[i];
        }
        
        std::vector<double> power_k = scaled_r;  // (D^{-1}A)^0 D^{-1}r
        z_local = power_k;
        
        std::vector<double> temp(n);
        for (int k = 1; k <= degree; ++k) {
            // power_k = D^{-1} A * power_{k-1}
            distributed_matvec(*A_ptr, power_k, temp, comm);
            for (int i = 0; i < n; ++i) {
                power_k[i] = diag_inv[i] * temp[i];
            }
            
            // z += power_k
            for (int i = 0; i < n; ++i) {
                z_local[i] += power_k[i];
            }
        }
        
    } else {  // CHEBYSHEV
        // Chebyshev polynomial evaluation via Horner's method
        std::vector<double> scaled_r(n);
        for (int i = 0; i < n; ++i) {
            scaled_r[i] = diag_inv[i] * r_local[i];
        }
        
        // p(A) r = c_0 r + c_1 Ar + c_2 A^2 r + ...
        std::vector<double> A_k_r = scaled_r;  // A^0 r
        for (int i = 0; i < n; ++i) {
            z_local[i] = coeffs[0] * A_k_r[i];
        }
        
        std::vector<double> temp(n);
        for (int k = 1; k <= degree; ++k) {
            distributed_matvec(*A_ptr, A_k_r, temp, comm);
            for (int i = 0; i < n; ++i) {
                A_k_r[i] = diag_inv[i] * temp[i];
            }
            
            for (int i = 0; i < n; ++i) {
                z_local[i] += coeffs[k] * A_k_r[i];
            }
        }
    }
}