# DeTecTUM — Decomposition Techniques with Thrifty Use of Memory
A lightweight linear algebra library offering a wide selection of matrix decompositions and solvers for real-valued dense matrices.

One of the distinctive traits of DeTecTUM is that all the algorithms are built around in-place operations, ensuring a minimal memory footprint. Fully implemented in plain C, the library adopts single-precision arithmetic and avoids dynamic memory allocation. All of these features also make DeTecTUM ideal for most embedded systems.

The library includes:
* Cholesky decomposition;
* LU decomposition;
* QR decomposition;
* bidiagonal decomposition;
* complete orthogonal decomposition (COD);
* singular value decomposition (SVD);
* Hessenberg decomposition;
* Schur decomposition;
* eigenvector computation;
* pseudo-eigenvector computation;
* solver for lower-triangular systems;
* solver for upper-triangular systems;
* solver for symmetric positive definite systems;
* solver for banded Hessenberg systems;
* solver for square full-rank systems;
* solver for general full-rank systems;
* solver for general rank-deficient systems;
* minimum-norm solver;
* Moore-Penrose pseudoinverse;
* matrix exponential;
* real matrix logarithm;
* real matrix square root.

## Testing on Linux
gcc -o mydemo demo/main_linsolve.c src/detectum.c -lm -I inc

## References
[1] Golub, Gene H., and Charles F. Van Loan. *Matrix computations*. JHU press, 2013.\
[2] Nash, John C. *Compact numerical methods for computers: linear algebra and function minimisation*. Routledge, 2018.\
[3] MATLAB documentation.\
[4] M. Galassi et al. *GNU Scientific Library Reference Manual (3rd Ed.)*. ISBN 0954612078.\
[5] Higham, Nicholas J. *Computing real square roots of a real matrix*. Linear Algebra and its applications 88 (1987): 405-430.\
[6] Higham, Nicholas J. *Functions of matrices: theory and computation*. Society for Industrial and Applied Mathematics, 2008.\
[7] Anderson, Edward. *Algorithm 978: Safe scaling in the level 1 BLAS*. ACM Transactions on Mathematical Software (TOMS) 44.1 (2017): 1-28.
