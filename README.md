# DeTecTUM — Decomposition Techniques with Thrifty Use of Memory
A lightweight linear algebra library offering a wide selection of matrix decompositions and solvers for real-valued dense matrices.

One of the distinctive traits of DeTecTUM is that all the algorithms are built around in-place operations, ensuring a minimal memory footprint. Fully implemented in plain C, the library adopts single-precision arithmetic and avoids dynamic memory allocation. All of these features also make DeTecTUM ideal for most embedded systems.

The library includes:
* Cholesky decomposition
* Aasen decomposition
* LU decomposition
* QR decomposition
* Bidiagonal decomposition
* Complete orthogonal decomposition (COD)
* Singular value decomposition (SVD)
* Hessenberg decomposition
* Schur decomposition
* Eigenvector computation
* Pseudo-eigenvector computation
* Solver for lower-triangular systems
* Solver for upper-triangular systems
* Solver for symmetric positive definite systems
* Solver for banded Hessenberg systems
* Solver for square full-rank systems
* Solver for general full-rank systems
* Solver for general rank-deficient systems
* Bounded-variable least-squares solver
* Minimum-norm solver
* Moore-Penrose pseudoinverse
* Matrix exponential
* Principal matrix logarithm
* Principal matrix square root

## Building Demos
To build a demo using GCC, run:
```
gcc -Wall -Wextra -Wpedantic -I inc src/detectum.c demo/<demo_name>.c -lm -o <demo_name>
```
Alternatively, run `make` alone to build the default demo (`main_linsolve`):
```
make
```
To build a different demo, specify its name using `TARGET`:
```
make TARGET=<demo_name>
```
Additional compiler flags can be passed through `EXTRA_CFLAGS`. For example, to enable optimization and vectorization diagnostics:
```
make TARGET=<demo_name> EXTRA_CFLAGS="-O3 -fopt-info-vec=vec.log"
```

## References
1. Golub, Gene H., and Charles F. Van Loan. "Matrix computations." JHU press, 2013.
2. Nash, John C. "Compact numerical methods for computers: linear algebra and function minimisation." Routledge, 2018.
3. M. Galassi et al. "GNU Scientific Library Reference Manual (3rd Ed.)." ISBN 0954612078.
4. Higham, Nicholas J. "Computing real square roots of a real matrix." Linear Algebra and its applications 88 (1987): 405-430.
5. Higham, Nicholas J. "Functions of matrices: theory and computation." Society for Industrial and Applied Mathematics, 2008.
6. Anderson, Edward. "Algorithm 978: Safe scaling in the level 1 BLAS." ACM Transactions on Mathematical Software (TOMS) 44.1 (2017): 1-28.
7. Lawson, Charles L., and Richard J. Hanson. "Solving least squares problems." Society for Industrial and Applied Mathematics, 1995.
8. Stark, Philip B., and Robert L. Parker. "Bounded-variable least-squares: an algorithm and applications." Computational Statistics 10 (1995): 129-129.
9. Moler, Cleve, and Charles Van Loan. "Nineteen dubious ways to compute the exponential of a matrix, twenty-five years later." SIAM review 45.1 (2003): 3-49.
