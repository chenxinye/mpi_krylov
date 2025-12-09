#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "matrix.hpp"
#include "preconditioner.hpp"
#include "jacobi.hpp"
#include "block_jacobi.hpp"
#include "ilu0.hpp"
#include "iluk.hpp"
#include "spai.hpp"
#include "utils.hpp"
#include "cg.hpp"
#include "bicgstab.hpp"
#include "gmres.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Build a small distributed 1D Poisson for demo (global N)
    int N = 100; if (argc>1) N = std::stoi(argv[1]);
    // simple block distribution
    int base = N / size; int rem = N % size;
    int local_n = base + (rank < rem ? 1 : 0);
    int row_offset = rank * base + std::min(rank, rem);

    CSRMatrix A;
    A.nrows = local_n; A.ncols = N; A.row_offset = row_offset;
    A.row_ptr.resize(local_n+1);
    std::vector<int> cols; cols.reserve(local_n*3);
    std::vector<double> vals; vals.reserve(local_n*3);
    A.row_ptr[0] = 0;
    for (int i = 0; i < local_n; ++i) {
        int gi = row_offset + i;
        if (gi-1 >= 0) { cols.push_back(gi-1); vals.push_back(-1.0); }
        cols.push_back(gi); vals.push_back(2.0);
        if (gi+1 < N) { cols.push_back(gi+1); vals.push_back(-1.0); }
        A.row_ptr[i+1] = (int)cols.size();
    }
    A.col_idx = std::move(cols);
    A.values = std::move(vals);

    // Build RHS b = 1
    std::vector<double> b_local(A.local_n(), 1.0);

    // initial x
    std::vector<double> x_local(A.local_n(), 0.0);

    // prepare preconditioners
    JacobiPrecond jac(A);
    BlockJacobiPrecond bjac(A, std::max(1, A.local_n()/4));
    ILU0Precond ilu0(A);
    ILUkPrecond iluk(A, 1);
    SPAIPrecond spai(A);

    // solvers to test
    struct Test { const char* name; int (*solve)(const CSRMatrix&, const std::vector<double>&, std::vector<double>&, int, double, MPI_Comm, Preconditioner*, int*, double*); Preconditioner* P; };
    // wrapper lambdas to adapt our solver signatures
    auto cg_wrapper = [](const CSRMatrix& A, const std::vector<double>& b, std::vector<double>& x, int maxit, double tol, MPI_Comm comm, Preconditioner* P, int* it, double* res){ return cg_solve(A,b,x,maxit,tol,comm,P,it,res); };
    auto bicg_wrapper = [](const CSRMatrix& A, const std::vector<double>& b, std::vector<double>& x, int maxit, double tol, MPI_Comm comm, Preconditioner* P, int* it, double* res){ return bicgstab_solve(A,b,x,maxit,tol,comm,P,it,res); };
    auto gmres_wrapper = [](const CSRMatrix& A, const std::vector<double>& b, std::vector<double>& x, int maxit, double tol, MPI_Comm comm, Preconditioner* P, int* it, double* res){ return gmres_solve(A,b,x,30,maxit,tol,comm,P,it,res); };

    std::vector<std::pair<std::string, Preconditioner*>> preconds = {
        {"none", nullptr},
        {"Jacobi", &jac},
        {"BlockJacobi", &bjac},
        {"ILU0", &ilu0},
        {"ILUk(k=1)", &iluk},
        {"SPAI", &spai}
    };

    struct SolverDesc {
    const char* name;
    std::function<int(const CSRMatrix&, const std::vector<double>&, std::vector<double>&,
                      int, double, MPI_Comm, Preconditioner*, int*, double*)> solver;
    };

    std::vector<SolverDesc> solvers = {
        {"CG", cg_wrapper},
        {"BiCGStab", bicg_wrapper},
        {"GMRES", gmres_wrapper}
    };

    if (rank==0) {
        std::cout << "Global N=" << N 
          << " ranks=" << size 
          << " local rows=" << A.nrows 
          << " global nnz=" << global_nnz(A, MPI_COMM_WORLD)
          << "\n";
        std::cout << "Testing solvers with preconditioners...\n";
    }

    for (auto &s : solvers) {
        for (auto &pc : preconds) {
            // reset x
            std::vector<double> x = x_local;
            int iters = -1; double final_norm = -1.0;
            double t0 = MPI_Wtime();
            int status = s.fn(A,b_local,x,1000,1e-8,MPI_COMM_WORLD, pc.second, &iters, &final_norm);
            double t1 = MPI_Wtime();
            if (rank==0) {
                std::cout << "Solver="<<s.name<<" Precond="<<pc.first<<" iters="<<iters<<" final_res="<<final_norm<<" time="<< (t1-t0) <<"s\n  ";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
