# DeTeGO - Decomposition Techniques Getting Optimal in terms of memory consumption
A linear algebra library offering a wide selection of matrix decompositions and direct solvers for real-valued dense matrices.

One of the distinctive traits of *DeTeGO* is that all the algorithms are built around in-place operations, ensuring a minimal memory footprint. Fully implemented in plain C, the library adopts single-precision arithmetic and avoids dynamic memory allocation. All of these features make *DeTeGO* ideal for embedded systems, too.

The library includes:
* Cholesky decomposition;
* LTL' decomposition;
* LU decomposition;
* QR decomposition;
* bidiagonalization;
* singular value decomposition (SVD);
* Hessenberg decomposition;
* Schur decomposition;
* eigenvector computation;
* pseudo-eigenvector computation;
* lower-triangular solver;
* upper-triangular solver;
* tridiagonal solver;
* symmetric positive definite solver;
* symmetric indefinite solver;
* full-rank square solver;
* full-rank least-squares solver;
* solver for underdetermined and overdetermined systems;  
* minimum-norm solver;
* Moore-Penrose pseudoinverse computation;
* matrix exponential computation;
* matrix exponentiation to positive integer power.

## Testing on Linux
gcc -o mydemo demo/mydemo.c src/detego.c -lm -I inc

## References
[1] Golub, Gene H., and Charles F. Van Loan. *Matrix computations*. JHU press, 2013.\
[2] Nash, John C. *Compact numerical methods for computers: linear algebra and function minimisation*. Routledge, 2018.\
[3] MATLAB documentation.
