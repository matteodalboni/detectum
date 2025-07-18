#ifndef DETECTUM_H
#define DETECTUM_H

#include <math.h>

typedef struct {
	int size[2];
	float* data;
} Matrixf;

// This macro allows accessing the element A(i, j).
#define at(A, i, j) ((A)->data[(i) + (j) * (A)->size[0]])

// This function returns the positive distance from abs(x) to the next 
// larger floating-point number.
static inline float epsf(float x)
{
	return powf(2.0f, floorf(log2f(fabsf(x))) - 23.0f);
}

// This function returns the 2-norm of the vector vec. The length of the
// vector is len, and stride is its increment.
static inline float normf(const float* vec, int len, int stride)
{
	int i;
	float x = 0, s = 0, t;

	for (i = 0; i < len; i++) {
		t = fabsf(vec[i * stride]);
		if (t > s) {
			s = t;
		}
	}
	if (s > 0) {
		for (i = 0; i < len; i++) {
			t = vec[i * stride] / s;
			x += t * t;
		}
		x = s * sqrtf(x);
	}
	return x;
}

// This function computes the Givens rotation pair (c,s) so that 
// [c -s; s c]*[a; b] = [r; 0].
static inline float givensf(float a, float b, float* c, float* s)
{
	float r = a;

	if (b == 0) {
		*c = 1;
		*s = 0;
	}
	else {
		r = hypotf(a, b);
		*c = +a / r;
		*s = -b / r;
	}
	return r;
}

#ifdef EOF // include stdio.h before detectum.h to enable this section
// This function prints the matrix A on the standard output according to
// the specified format.
static inline void matrixf_print(Matrixf* A, const char* format)
{
	int i, j;

	for (i = 0; i < A->size[0]; i++) {
		for (j = 0; j < A->size[1]; j++) {
			printf(format, at(A, i, j));
		}
		printf("\n");
	}
}
#endif

#ifdef RAND_MAX // include stdlib.h before detectum.h to enable this section
// This function initializes a rows-by-cols matrix instance while dynamically
// allocating its data memory. On allocation failure, the data pointer is null.
static inline Matrixf matrixf(int rows, int cols)
{
	Matrixf A = { { rows, cols }, calloc(sizeof(float), rows * cols) };
	return A;
}
#endif

// This function initializes the rows-by-cols matrix A. The array data
// collects the matrix elements, which must be stored in column-major order.
// If the data are initially arranged in row-major layout, the flag ordmem 
// can be raised to enable the conversion to column-major memory order.
void matrixf_init(Matrixf* A, int rows, int cols, float* data, int ordmem);

// This function permutes the m-by-n matrix A according to the vector of 
// permutation indices p, which encodes the permutation matrix P: if p is m-by-1,
// the rows of A are permuted, whereas, if p is 1-by-n, the columns of A are 
// permuted. If the flag reverse is enabled, the function transforms also the 
// vector of permutation indices so that, if p encodes a column permutation:
// - if p is 1-by-n and reverse == 0, A is transformed into A*P;
// - if p is 1-by-m and reverse != 0, A is transformed into P*A and p transformed.
// Whereas, if if p encodes a row permutation:
// - if p is m-by-1 and reverse == 0, A is transformed into P*A;
// - if p is n-by-1 and reverse != 0, A is transformed into A*P and p transformed.
// Essentially, the flag allows reversing the multiplication order, transforming a
// column permutation into a row permutation, or vice versa. This enables switching 
// between pre- and post-multiplication by the same permutation matrix P.
// On size mismatch, the function returns -1. On success, it returns 0.
int matrixf_permute(Matrixf* A, Matrixf* p, int reverse);

// This function transposes in place the input matrix.
void matrixf_transpose(Matrixf* A);

// This function performs the Cholesky decomposition of the square symmetric 
// positive definite matrix A such that A = R'*R. The upper triangular part of A 
// is overwritten by R. The function returns -1 if A is not square, -2 if A is 
// not positive definite. On success, it returns 0.
int matrixf_decomp_chol(Matrixf* A);

// This function performs the LTL' decomposition with pivoting using the method
// of Aasen. The n-by-n symmetric indefinite matrix A is decomposed so that  
// P*A*P' = L*T*L', where L is unit lower triangular, T is symmetric tridiagonal,
// and P is a permutation matrix. The matrix A is transformed so that:
// 1) its diagonal contains the main diagonal of T;
// 2) its first diagonal above the main one stores the first diagonals of T;
// 3) its strictly lower part holds the strictly lower part of L, assuming also
//    L(i,0) = 0 for 0 < i < n;
// 4) since L(i,0) = 0 for 0 < i < n, the permutation matrix P is encoded in the 
//    first column of A so that P(0,0) = 1 and P(i,A(i,0)) = 1 for 0 < i < n.
// The function returns -1 if A is not square. On success, it returns 0.
int matrixf_decomp_ltl(Matrixf* A);

// This function performs the LU decomposition with partial pivoting of the 
// n-by-n matrix A so that A = P'*L*U. The matrix A is transformed so that 
// its upper triangular part stores the matrix U, whereas its strictly lower
// triangular part contains the matrix L, assuming that all the entries of 
// the main diagonal of L are ones (unit lower triangular matrix). B must 
// have n rows. At output, the rows of B are permuted so that B is 
// transformed into P*B: if B is initialized as an n-by-n identity matrix, 
// it is transformed into P. On size mismatch or non-square matrix, the 
// function returns -1. On success, it returns 0.
int matrixf_decomp_lu(Matrixf* A, Matrixf* B);

// This function performs the LU decomposition with partial pivoting of the 
// banded Hessenberg matrix A. In particular, A must be a Hessenberg matrix 
// with upper bandwidth ubw >= 0. For instance, if ubw = 1, A is tridiagonal.
// The matrix A is transformed so that its upper triangular part stores the 
// matrix U, whereas its first subdiagonal encodes the transformations that 
// are needed to assemble the inverse of the permuted lower triangular matrix
// L. If A is not square or ubw < 0, the function returns -1. On success, it 
// returns 0.
int matrixf_decomp_lu_banded(Matrixf* A, int ubw);

// This function unpacks the compact form of LU decomposition of the banded 
// Hessenberg matrix A. In particular, the function accumulates the inverse of
// the permuted lower triangular matrix L onto B, transforming B into inv(L)*B.
// Matrix A remains the same. For instance, this function enables the solution
// to the linear system U*x = inv(L)*B, determining its right-hand side. On 
// size mismatch, the function returns -1. On success, it returns 0.
int matrixf_unpack_lu_banded(Matrixf* A, Matrixf* B);

// This function performs the QR decomposition of the m-by-n matrix A, which is
// transformed into matrix R. If Q is a null pointer, the computation of 
// the orthogonal matrix is omitted, and the strictly lower triangular part of A 
// stores the relevant parts of the Householder vectors, which can be used to
// accumulate Q afterwards. For the sake of clarity, the relevant part of a 
// Householder vector comprises all elements but the first, which is always 1. 
// If P is a non-null pointer, the decomposition makes use of column pivoting 
// so that A*P = Q*R and so that the magnitude of the elements of the main 
// diagonal of R is decreasing. If P is initialized as a 1-by-n vector and n > 1,
// the permutations are encoded so that (P(i),i) for 0 <= i < n are the unit 
// elements of the permutation matrix. Else, if P is initialized as an n-by-n 
// matrix, the full permutation matrix is returned. Additionally, one can provide
// the matrix B, which is transformed into Q'*B; this computation is skipped if B
// is a null pointer. If m > n, the function can also produce the economy-size 
// decomposition such that only the first n columns of Q are computed (thin Q) 
// and the last m - n rows of R are excluded so that R becomes n-by-n. To enable
// the economy-size decomposition, Q must be initialized as an m-by-n matrix. On
// size mismatch, the function returns -1. On success, it returns 0.
int matrixf_decomp_qr(Matrixf* A, Matrixf* Q, Matrixf* P, Matrixf* B);

// This function unpacks an orthogonal matrix from its factored representation. 
// In particular, the function transforms B into Q'*B by forward accumulation 
// of Householder matrices. Matrix A remains the same. The lower triangular part
// of A below the s-th subdiagonal must store the relevant parts of the Householder
// vectors. For instance, if s = 0, the Householder vectors are below the main 
// diagonal of A; whereas, if s = 1, the Householder vectors are below the first 
// subdiagonal of A. On size mismatch or if s < 0, the function returns -1. On 
// success, it returns 0.
int matrixf_unpack_householder_fwd(Matrixf* A, Matrixf* B, int s);

// This function unpacks an orthogonal matrix from its factored representation. 
// In particular, the function transforms B into Q*B by backward accumulation 
// of Householder matrices. Matrix A remains the same. The lower triangular part
// of A below the s-th subdiagonal must store the relevant parts of the Householder
// vectors. For instance, if s = 0, the Householder vectors are below the main 
// diagonal of A; whereas, if s = 1, the Householder vectors are below the first 
// subdiagonal of A. On size mismatch or if s < 0, the function returns -1. On 
// success, it returns 0.
int matrixf_unpack_householder_bwd(Matrixf* A, Matrixf* B, int s);

// This function accomplishes the bidiagonalization of the m-by-n matrix A so
// that A = U*B*V'. A is overwritten by the bidiagonal matrix B. Specifically, B
// is upper bidiagonal if m >= n, otherwise B is lower bidiagonal. The computation
// of the matrices U and V is omitted if these are null pointers. If U (V) is
// a null pointer, the relevant parts of the Householder vectors are stored
// off the bidiagonal band and can be used to accumulate U (V) afterwards.
// In particular, U is encoded below the main diagonal of A, while V is encoded 
// below the first subdiagonal of A'.
// The function can also produce the economy-size decomposition such that:
// - if m > n, only the first n columns of U are computed (thin U) and the last 
//   m - n rows of B are excluded so that B becomes n-by-n; to enable the 
//   economy-size decomposition, U must be initialized as an m-by-n matrix;
// - if m < n, only the first m columns of V are computed (thin V) and the last 
//   n - m columns of B are excluded so that B becomes m-by-m; to enable the 
//   economy-size decomposition, V must be initialized as an n-by-m matrix;
// - if m = n, the economy-size decomposition is the same as the full one.
// On size mismatch, the function returns -1. On success, it returns 0.
int matrixf_decomp_bidiag(Matrixf* A, Matrixf* U, Matrixf* V);

// This function performs the complete orthogonal decomposition of the m-by-n matrix
// A so that A*P = U*T*V'; T = [L, 0; 0; 0], L being a lower triangular square block
// whose size is r-by-r, where r is the rank of A. The permutation matrix P must be
// always provided, as the function makes use of QR decomposition with column 
// pivoting. If P is initialized as a 1-by-n vector and n > 1, the permutations are
// encoded so that (P(i),i) for 0 <= i < n are the unit elements of the permutation 
// matrix. Else, if P is initialized as an n-by-n matrix, the full permutation 
// matrix is returned. The computation of the matrices U and V is omitted if these
// are null pointers. If m > n, the function can also produce the economy-size 
// decomposition such that only the first n columns of U are computed (thin U) and
// the last m - n rows of T are excluded so that T becomes n-by-n. To enable the 
// economy-size decomposition, U must be initialized as an m-by-n matrix. tol is the
// tolerance to determine the rank of A: if the input tolerance is negative, the 
// default value max(m,n)*eps(R(0,0)) is used instead, where R(0,0) is the 
// on-diagonal element of R with the largest magnitude, R being the upper triangular
// matrix obtained by QR decomposition with column pivoting of A. On size mismatch,
// the function returns -1. On success, it returns the rank of A.
int matrixf_decomp_cod(Matrixf* A, Matrixf* P, Matrixf* U, Matrixf* V, float tol);

// This function performs the singular value decomposition of the m-by-n matrix A
// by QR iteration. The decomposition is such that A = U*S*V'. The computation of 
// the matrix of the left singular vectors (i.e. matrix U) is performed only if U 
// is initialized as an m-by-m matrix, whereas it is omitted if U is a null pointer.
// Likewise, the computation of the matrix of the right singular vectors (i.e. matrix
// V) is performed only if V is initialized as an n-by-n matrix, whereas it is
// omitted if V is a null pointer. The function can also produce the economy-size 
// decomposition such that:
// - if m > n, only the first n columns of U are computed (thin U) and the last 
//   m - n rows of S are excluded so that S becomes n-by-n; to enable the 
//   economy-size decomposition, U must be initialized as an m-by-n matrix;
// - if m < n, only the first m columns of V are computed (thin V) and the last 
//   n - m columns of S are excluded so that S becomes m-by-m; to enable the 
//   economy-size decomposition, V must be initialized as an n-by-m matrix;
// - if m = n, the economy-size decomposition is the same as the full one.
// On size mismatch, the function returns -1. It returns -2 if the maximum number
// of sweeps is reached. On success, it returns the number of sweeps performed.
int matrixf_decomp_svd(Matrixf* A, Matrixf* U, Matrixf* V);

// This function performs the singular value decomposition of the m-by-n matrix A
// by one-sided Jacobi algorithm. The decomposition is such that A = U*S*V'. If U
// is a null pointer, the matrix A is overwritten by U*S: in this case the singular
// values are the norms of A = U*S columns. Whereas, if U is initialized as an 
// m-by-m matrix, the matrix of the left singular vectors (i.e. matrix U) is 
// explicitly computed, and A is overwritten by the diagonal matrix S. 
// The computation of the matrix of the right singular vectors (i.e. matrix V) 
// is performed only if V is initialized as an n-by-n matrix, whereas it is 
// omitted if V is a null pointer. The function can also produce the economy-size 
// decomposition such that:
// - if m > n, only the first n columns of U are computed (thin U) and the last 
//   m - n rows of S are excluded so that S becomes n-by-n; to enable the 
//   economy-size decomposition, U must be initialized as an m-by-n matrix;
// - if m < n, only the first m columns of V are computed (thin V) and the last 
//   n - m columns of S are excluded so that S becomes m-by-m; to enable the 
//   economy-size decomposition, V must be initialized as an n-by-m matrix;
// - if m = n, the economy-size decomposition is the same as the full one.
// The function returns -1 on size mismatch. It returns -2 if the maximum number
// of sweeps is reached. On success, it returns the number of sweeps performed.
int matrixf_decomp_svd_jacobi(Matrixf* A, Matrixf* U, Matrixf* V);

// This function performs the Hessenberg decomposition of the square matrix A, 
// which is transformed into the Hessenberg matrix H so that A = P*H*P'. If P 
// is a null pointer, the computation of the orthogonal matrix is omitted, and
// the triangular part of A below the first subdiagonal stores the relevant 
// parts of the Householder vectors, which can be used to accumulate P afterwards. 
// The function returns -1 if the matrices are not square or on size mismatch.
// On success, it returns 0.
int matrixf_decomp_hess(Matrixf* A, Matrixf* P);

// This function performs the Schur decomposition of the symmetric matrix A so that
// A = U*D*U'. A is transformed into the diagonal matrix D whose main diagonal holds
// the eigenvalues of A. If U is a null pointer, the computation of the orthogonal 
// matrix is omitted. The function returns -1 if the matrices are not square or on 
// size mismatch. It returns -2 if the maximum number of sweeps is reached. On 
// success, it returns the number of sweeps performed.
int matrixf_decomp_schur_symm(Matrixf* A, Matrixf* U);

// This function performs the Schur decomposition of the square matrix A so that
// A = U*T*U'. A is transformed into the quasitriangular matrix T whose main diagonal
// holds the real eigenvalues, whereas the complex eigenvalues are expressed as
// 2-by-2 blocks along the main diagonal. In particular, each 2-by-2 block has the
// form B = [x, y; z, x], and its complex conjugate eigenvalues, i.e. x+sqrt(-y*z)*i 
// and x-sqrt(-y*z)*i are also a complex conjugate pair of eigenvalues of A. 
// If U is a null pointer, the computation of the orthogonal matrix is omitted. The 
// function returns -1 if the matrices are not square or on size mismatch. It returns
// -2 if the maximum number of sweeps is reached. On success, it returns the number 
// of sweeps performed.
int matrixf_decomp_schur(Matrixf* A, Matrixf* U);

// This function computes the matrices of right (V) and left (W) eigenvectors
// from the quasitriangular matrix T and orthogonal matrix U obtained by Schur 
// decomposition. Matrices T and U remain the same. If the flag pseudo is enabled,
// the function calculates the pseudo-eigenvectors; else, it calculates the 
// eigenvectors. If the k-th and (k+1)-th eigenvalues are a complex conjugate pair,
// since a complex conjugate pair of eigenvalues has complex conjugate eigenvectors, 
// the function computes only the right and left eigenvectors corresponding to the 
// eigenvalue with positive imaginary part: the real and imaginary parts of the 
// computed eigenvectors are stored in the k-th and (k+1)-th columns of V and W, 
// respectively. If V (or W) is a null pointer, its computation is omitted. The 
// array work is the additional workspace memory: in general, if T is n-by-n, its 
// minimum length is max(n^2,4*(n-2)^2). If all the eigenvalues are real, the 
// minimum length of work is n^2. The function returns -1 if the matrices of 
// eigenvectors are not n-by-n; it returns -2 if a singularity is detected. On 
// success, it returns 0.
int matrixf_get_eigenvectors(Matrixf* T, Matrixf* U,
	Matrixf* V, Matrixf* W, int pseudo, float* work);

// This function performs the forward substitution on the lower-triangular
// matrix L to solve the the system L*X = B for X. The flag unitri indicates
// whether L is unitriangular: if the flag is enabled, the elements on the main
// diagonal are ignored. Matrix L remains the same while B is destroyed. Also, 
// the matrix B can share the data array with X, provided that the array is 
// large enough to accommodate the larger of B or X. It returns -2 if L is rank
// deficient, -1 on size mismatch. On success, it returns 0.
int matrixf_solve_tril(Matrixf* L, Matrixf* B, Matrixf* X, int unitri);

// This function performs the backward substitution on the upper-triangular
// matrix U to solve the the system U*X = B for X. The flag unitri indicates
// whether U is unitriangular: if the flag is enabled, the elements on the main
// diagonal are ignored. Matrix U remains the same while B is destroyed. Also,
// the matrix B can share the data array with X, provided that the array is 
// large enough to accommodate the larger of B or X. It returns -2 if U is rank
// deficient, -1 on size mismatch. On success, it returns 0.
int matrixf_solve_triu(Matrixf* U, Matrixf* B, Matrixf* X, int unitri);

// This function solves in place the linear system A*X = B for X by Cholesky
// decomposition of the square symmetric positive definite matrix A. The upper 
// triangular part of A is transformed according to Cholesky decomposition
// while B is overwritten with the matrix X. If A is not positive definite, the 
// function returns -2. On size mismatch or non-square system, the function 
// returns -1. On success, it returns 0.
int matrixf_solve_chol(Matrixf* A, Matrixf* B);

// This function solves in place the linear system A*X = B for X by LTL'
// decomposition with pivoting of the square symmetric indefinite matrix A. 
// The matrix A is destroyed while B is overwritten with the matrix X. If A is 
// singular, the function returns -2. On size mismatch or non-square system,
// the function returns -1. On success, it returns 0.
int matrixf_solve_ltl(Matrixf* A, Matrixf* B);

// This function solves in place the linear system A*X = B for X by LU
// decomposition with partial pivoting of the square matrix A. The matrix A
// is transformed according to LU decomposition while B is overwritten with the 
// matrix X. If A is singular, the function returns -2. On size mismatch or 
// non-square system, the function returns -1. On success, it returns 0.
int matrixf_solve_lu(Matrixf* A, Matrixf* B);

// This function solves in place the linear system A*X = B for X by LU
// decomposition with partial pivoting of the banded Hessenberg matrix A. In 
// particular, A must be a Hessenberg matrix with upper bandwidth ubw >= 0. For
// instance, if ubw = 1, A is tridiagonal. Matrix A is destroyed, whereas B is 
// overwritten with the matrix X. If A is singular, the function returns -2. On
// size mismatch, non-square system or ubw < 0, the function returns -1. On 
// success, it returns 0.
int matrixf_solve_lu_banded(Matrixf* A, Matrixf* B, int ubw);

// This function solves the linear system A*X = B by QR decomposition
// of the full-rank matrix A. The returned solution is minimum-norm. The 
// matrices A and B are destroyed. The matrix B can share the data array with X,
// provided that the array is large enough to accommodate the larger of B or X.
// If A is rank deficient, the function returns -2. On size mismatch, the 
// function returns -1. On success, it returns 0.
int matrixf_solve_qr(Matrixf* A, Matrixf* B, Matrixf* X);

// This function solves the linear system A*X = B for X by QR decomposition 
// with column pivoting. The matrices A and B are destroyed. The matrix B
// can share the data array with X, provided that the array is large enough to 
// accommodate the larger of B or X. tol is the tolerance to determine the rank 
// of the A m-by-n matrix A: if the input tolerance is negative, the default 
// value max(m,n)*eps(R(0,0)) is used instead, where R(0,0) is the on-diagonal 
// element of R with the largest magnitude, R being the upper triangular matrix 
// obtained by the QR decomposition with column pivoting of A. The array work is
// the additional workspace memory: its minimum length is n. On size mismatch, 
// the function returns -1. On success, it returns 0.
int matrixf_solve_qrp(Matrixf* A, Matrixf* B, Matrixf* X, float tol, float* work);

// This function solves the linear system A*X = B for X by complete orthogonal
// decomposition of the m-by-n matrix A: this ensures that X be the minimum-norm
// solution. The matrices A and B are destroyed. The matrix B can share the data
// array with X, provided that the array is large enough to accommodate the larger
// of B or X. tol is the tolerance to determine the rank of A: if the input 
// tolerance is negative, the default value max(m,n)*eps(R(0,0)) is used instead, 
// where R(0,0) is the on-diagonal element of R with the largest magnitude, R being
// the upper triangular matrix obtained by QR decomposition with column pivoting
// of A. The array work is the additional workspace memory: its minimum length is n.
// On size mismatch, the function returns -1. On success, it returns 0.
int matrixf_solve_cod(Matrixf* A, Matrixf* B, Matrixf* X, float tol, float* work);

// This function substitutes the m-by-n input matrix A with its Moore-Penrose
// pseudoinverse obtained by Jacobi SVD. tol is the tolerance to treat singular
// values as zero: if the input tolerance is negative, the default value 
// max(m,n)*eps(S(0,0)) is used instead, where S(0,0) is the maximum singular 
// value of A. The array work is the additional workspace memory: its minimum 
// length is min(m,n)*(min(m,n)+1). Also, the function returns the number of 
// sweeps performed.
int matrixf_pseudoinv(Matrixf* A, float tol, float* work);

// This function computes the exponential of the square matrix A by scaling
// and squaring algorithm with Pad� approximation. The array work is the
// additional workspace memory: if A is n-by-n, its minimum length is n*(3*n+1).
// The function returns -1 if the input matrix is not square; it returns -2 if a
// singularity is detected. On success, it returns 0.
int matrixf_exp(Matrixf* A, float* work);

// This function computes the principal real square root of a real matrix by real
// Schur method. The n-by-n matrix A is transformed into X so that A = X*X. The 
// array work is the additional workspace memory: its minimum length is n*(n+1).
// If Schur decomposition fails to converge, the function returns -2. If A has 
// negative real eigenvalues, a real square root does not exist, and the function
// returns -3. If A is singular and its square root cannot be computed, the 
// function returns -4: in fact, a singular matrix may not have a square root. If
// A is not square, the function returns -1. On success, it returns the number of 
// sweeps performed by Schur decomposition.
int matrixf_sqrt(Matrixf* A, float* work);

// This function performs the general matrix multiplication (GEMM), which has
// the form C = alpha*op(A)*op(B) + beta*C, where A, B and C are general 
// matrices, alpha and beta are scalars, and op(X) is one of X or its
// transpose. Therefore, if transX = 0, op(X) = X, else op(X) = X', X' being 
// the transpose of X. C must be a distinct instance with respect to A and B.
// Matrices A and B remain the same. On size mismatch, the function returns -1.
// On success, it returns 0.
int matrixf_multiply(Matrixf* A, Matrixf* B, Matrixf* C,
	float alpha, float beta, int transA, int transB);

#endif
