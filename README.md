# mpi_krylov: Krylov Solver using MPI

This project provides parallel implementations of Krylov subspace solvers for sparse linear systems using MPI in C++. The solvers include Conjugate Gradient (CG), BiCGStab, GMRES, and communication avoid GMRES, with optional preconditioners such as Jacobi, Block Jacobi, and ILU0. 



## Setup and Installation

Ensure you have a C++17 compatible compiler and an MPI implementation installed, such as OpenMPI or MPICH. On most systems, MPI can be installed via the package manager.  For example, in mac, we use ``brew install open-mpi``.

To compile the project, navigate to the project root directory and run:

```bash
make
```

This will build the executable named solver. To clean compiled object files and the executable, use:

```bash
make clean
```

## Running the Program

Since this is an MPI program, use mpirun or mpiexec to run the executable. For example, to run with four MPI processes:

```bash
mpirun -np 4 ./solver
```

The program performs on a small poisson dataset (can be replaced), executes all three solvers on a test sparse matrix and prints iteration counts, final residuals, and runtime for each solver. Preconditioners, if included, are applied automatically. Adjust the number of MPI processes using -np according to your system capabilities.


In my mac:
```text
mpi_krylov % mpirun -np 2 ./solver
Solver=CG iters=4 final_res=2.8794e-12 time=0.041735 s
Solver=BiCGStab iters=4 final_res=5.01607e-07 time=0.055475 s
Solver=GMRES iters=4 final_res=8.71163e-08 time=0.039712 s
mpi_krylov % mpirun -np 4 ./solver
Solver=CG iters=6 final_res=8.52445e-11 time=0.027062 s
Solver=BiCGStab iters=6 final_res=4.13439e-07 time=0.027559 s
Solver=GMRES iters=6 final_res=8.94959e-08 time=0.017409 s
```
