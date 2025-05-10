#ifndef DETEGO_H
#define DETEGO_H

#include <math.h>

typedef struct {
	int size[2];
	float* data;
} Matrixf;

#ifndef DETEGO_SVD_SWEEPMAX
#define DETEGO_SVD_SWEEPMAX (10000)
#endif
#ifndef DETEGO_SVD_TOL
#define DETEGO_SVD_TOL (3e-7f)
#endif
#ifndef DETEGO_SVD_JACOBI_SWEEPMAX
#define DETEGO_SVD_JACOBI_SWEEPMAX (1000)
#endif
#ifndef DETEGO_SVD_JACOBI_TOL
#define DETEGO_SVD_JACOBI_TOL (3e-7f)
#endif
#ifndef DETEGO_SCHUR_SYMM_SWEEPMAX
#define DETEGO_SCHUR_SYMM_SWEEPMAX (10000)
#endif
#ifndef DETEGO_SCHUR_SYMM_TOL
#define DETEGO_SCHUR_SYMM_TOL (3e-7f)
#endif
#ifndef DETEGO_SCHUR_SWEEPMAX
#define DETEGO_SCHUR_SWEEPMAX (10000)
#endif
#ifndef DETEGO_SCHUR_TOL
#define DETEGO_SCHUR_TOL (3e-7f)
#endif
#ifndef DETEGO_SCHUR_AD_HOC_SHIFT_COUNT
#define DETEGO_SCHUR_AD_HOC_SHIFT_COUNT (5)
#endif
#ifndef DETEGO_EXPM_PADE_ORDER
#define DETEGO_EXPM_PADE_ORDER (4)
#endif

// This macro allows accessing the element A(i, j).
#define at(A, i, j) ((A)->data[(i) + (j) * (A)->size[0]])

// This function returns the positive distance from abs(x) to the next 
// larger floating-point number.
static inline float epsf(const float x)
{
	return powf(2.0f, floorf(log2f(fabsf(x))) - 23.0f);
}

// This function returns the 2-norm of the vector vec. The length of the
// vector is len, and stride is its increment.
static inline float normf(const float* vec, const int len, const int stride)
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
static inline void givensf(const float a, const float b, float* c, float* s)
{
	float tau = 0;

	if (b == 0) {
		*c = 1;
		*s = 0;
	}
	else if (fabsf(b) > fabsf(a)) {
		tau = -a / b;
		*s = 1.0f / sqrtf(1 + tau * tau);
		*c = *s * tau;
	}
	else {
		tau = -b / a;
		*c = 1.0f / sqrtf(1 + tau * tau);
		*s = *c * tau;
	}
}

#ifdef EOF // include stdio.h before detego.h to enable this section
// This function prints the matrix A on the standard output according to
// the specified format.
static inline void matrixf_print(Matrixf* A, char* format)
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

#ifdef RAND_MAX // include stdlib.h before detego.h to enable this section
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
void matrixf_init(Matrixf* A, int rows, int cols, float* data, const int ordmem);

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
// On size mismatch, the function returns -1.
int matrixf_permute(Matrixf* A, Matrixf* p, const int reverse);

// This function transposes in place the input matrix.
void matrixf_transpose(Matrixf* A);

// This function performs the Cholesky decomposition of the square 
// symmetric positive definite matrix A such that A = R'*R. The upper 
// triangular part of A is overwritten by R. The function returns -1
// if A is not square, 1 if A is not positive definite.
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
// The function returns -1 if A is not square.
int matrixf_decomp_ltl(Matrixf* A);

// This function performs the LU decomposition with partial pivoting of the 
// square matrix A. The matrix A is transformed so that its upper triangular
// part stores the matrix U, whereas its strictly lower triangular part 
// contains the matrix L, assuming that all the entries of the main diagonal
// of L are ones (unit lower triangular matrix). P is a matrix with as many 
// rows as A has. At output, the rows of the input matrix P are permuted: if
// P is initialized as an identity matrix, its rows are permuted so that 
// A = P'*L*U. On size mismatch or non-square matrix, the function returns -1.
int matrixf_decomp_lu(Matrixf* A, Matrixf* P);

// This function performs the LU decomposition with partial pivoting of the 
// banded Hessenberg matrix A. In particular, A must be a Hessenberg matrix 
// with upper bandwidth ubw >= 0. For instance, if ubw = 1, A is tridiagonal.
// The matrix A is transformed so that its upper triangular part stores the 
// matrix U, whereas its first subdiagonal encodes the transformations that 
// are needed to assemble the inverse of the permuted lower triangular matrix
// L. If A is not square or ubw < 0, the function returns -1.
int matrixf_decomp_lu_banded(Matrixf* A, const int ubw);

// This function unpacks the compact form of LU decomposition of the banded 
// Hessenberg matrix A. In particular, the function accumulates the inverse of
// the permuted lower triangular matrix L onto B, transforming B into inv(L)*B.
// For instance, this function enables the solution of the linear system 
// U*x = inv(L)*B, determining its right-hand side. On size mismatch, the 
// function returns -1.
int matrixf_unpack_lu_banded(Matrixf* A, Matrixf* B);

// This function performs the QR decomposition of the m-by-n matrix A, which is
// transformed into matrix R. If Q is a null pointer, the computation of 
// the orthogonal matrix is omitted, and the strictly lower triangular part of A 
// stores the relevant parts of the Householder vectors, which can be used to
// accumulate Q afterwards. If P is a non-null pointer, the decomposition makes 
// use of column pivoting so that A*P = Q*R and so that the magnitude of the 
// elements of the main diagonal of R is decreasing. If P is initialized as a 
// 1-by-n vector and n > 1, the permutations are encoded so that (P(i),i) for 
// 0 <= i < n are the unit elements of the permutation matrix. Else, if P is 
// initialized as an n-by-n matrix, the full permutation matrix is returned. 
// Additionally, one can provide the matrix B, which is transformed into Q'*B; 
// this computation is skipped if B is a null pointer. If m > n, the function 
// can also produce the economy-size decomposition such that only the first n
// columns of Q are computed (thin Q) and the last m - n rows of R are excluded
// so that R becomes n-by-n. To enable the economy-size decomposition, Q must be 
// initialized as an m-by-n matrix. On size mismatch, the function returns -1.
int matrixf_decomp_qr(Matrixf* A, Matrixf* Q, Matrixf* P, Matrixf* B);

// This function unpacks an orthogonal matrix from its factored representation. 
// In particular, the function transforms B into Q'*B by forward accumulation 
// of Householder matrices. The lower triangular part of A below the s-th 
// subdiagonal stores the relevant parts of the Householder vectors. For
// instance, if s = 0, the Householder vectors are below the main diagonal of 
// A; whereas, if s = 1, the Householder vectors are below the first subdiagonal
// of A. On size mismatch or if s < 0, the function returns -1.
int matrixf_unpack_householder_fwd(Matrixf* A, Matrixf* B, const int s);

// This function unpacks an orthogonal matrix from its factored representation. 
// In particular, the function transforms B into Q*B by backward accumulation 
// of Householder matrices. The lower triangular part of A below the s-th 
// subdiagonal stores the relevant parts of the Householder vectors. For
// instance, if s = 0, the Householder vectors are below the main diagonal of 
// A; whereas, if s = 1, the Householder vectors are below the first subdiagonal
// of A. On size mismatch or if s < 0, the function returns -1.
int matrixf_unpack_householder_bwd(Matrixf* A, Matrixf* B, const int s);

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
// On size mismatch, the function returns -1.
int matrixf_decomp_bidiag(Matrixf* A, Matrixf* U, Matrixf* V);

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
// On size mismatch, the function returns -1. Otherwise, it returns the number
// of iterations performed.
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
// The function returns -1 on size mismatch. Otherwise, it returns the number
// of iterations performed.
int matrixf_decomp_svd_jacobi(Matrixf* A, Matrixf* U, Matrixf* V);

// This function performs the Hessenberg decomposition of the square matrix A, 
// which is transformed into the Hessenberg matrix H so that A = P*H*P'. If P 
// is a null pointer, the computation of the orthogonal matrix is omitted, and
// the triangular part of A below the first subdiagonal stores the relevant 
// parts of the Householder vectors, which can be used to accumulate P afterwards. 
// The function returns -1 if the matrices are not square or on size mismatch.
int matrixf_decomp_hess(Matrixf* A, Matrixf* P);

// This function performs the Schur decomposition of the symmetric matrix A so that
// A = U*D*U'. A is transformed into the diagonal matrix D whose main diagonal holds
// the eigenvalues of A. If U is a null pointer, the computation of the orthogonal 
// matrix is omitted. The function returns -1 if the matrices are not square or on 
// size mismatch. Otherwise, it returns the number of iterations performed.
int matrixf_decomp_schur_symm(Matrixf* A, Matrixf* U);

// This function performs the Schur decomposition of the square matrix A so that
// A = U*T*U'. A is transformed into the quasitriangular matrix T whose main diagonal
// holds the real eigenvalues, whereas the complex eigenvalues are expressed as
// 2-by-2 blocks along the main diagonal. In particular, each 2-by-2 block has the
// form B = [x, y; z, x], and its complex conjugate eigenvalues, i.e. x+sqrt(-y*z)*i 
// and x-sqrt(-y*z)*i are also a complex conjugate pair of eigenvalues of A. 
// If U is a null pointer, the computation of the orthogonal matrix is omitted. The 
// function returns -1 if the matrices are not square or on size mismatch. Otherwise,
// it returns the number of iterations performed.
int matrixf_decomp_schur(Matrixf* A, Matrixf* U);

// This function computes the right (V) and left (W) eigenvectors starting from the
// quasitriangular matrix T and orthogonal matrix U obtained by Schur decomposition.
// If the flag pseudo is enabled, the function returns the pseudo-eigenvectors; else,
// it returns the eigenvectors. If the k-th and (k+1)-th eigenvalues are a complex 
// conjugate pair, since a complex conjugate pair of eigenvalues has complex conjugate
// eigenvectors, the function computes only the right and left eigenvectors 
// corresponding to the eigenvalue with positive imaginary part: the real and imaginary
// parts of the computed eigenvectors are stored in the k-th and (k+1)-th columns of V
// and W, respectively. If V (or W) is a null pointer, its computation is omitted.
// The array work is the additional workspace memory: in general, if T is n-by-n, its 
// minimum length is max(n^2,4*(n-2)^2). If all the eigenvalues are real, the minimum
// length of work is n^2. The function returns -1 if the matrices of eigenvectors are
// not n-by-n; it returns 1 if the matrix T is defective to working precision.
int matrixf_get_eigenvectors(Matrixf* T, Matrixf* U,
	Matrixf* V, Matrixf* W, int pseudo, float* work);

// This function performs the forward substitution on the lower-triangular
// matrix L to solve the the system L*X = B for X. The flag unitri indicates
// whether L is unitriangular: if the flag is enabled, the elements on the main
// diagonal are ignored. The matrix B is destroyed. Also, the matrix B can
// share the data array with X, provided that the array is large enough to 
// accommodate the larger of B or X. It returns 1 if L is rank deficient,
// -1 on size mismatch.
int matrixf_solve_tril(Matrixf* L, Matrixf* B, Matrixf* X, int unitri);

// This function performs the backward substitution on the upper-triangular
// matrix U to solve the the system U*X = B for X. The flag unitri indicates
// whether U is unitriangular: if the flag is enabled, the elements on the main
// diagonal are ignored. The matrix B is destroyed. Also, the matrix B can
// share the data array with X, provided that the array is large enough to 
// accommodate the larger of B or X. It returns 1 if U is rank deficient,
// -1 on size mismatch.
int matrixf_solve_triu(Matrixf* U, Matrixf* B, Matrixf* X, int unitri);

// This function solves in place the linear system A*X = B for X by Cholesky
// decomposition of the square symmetric positive definite matrix A. The upper 
// triangular part of A is transformed according to Cholesky decomposition
// while B is overwritten with the matrix X. If A is not positive definite, the 
// function returns 1. On size mismatch or non-square system, the function 
// returns -1.
int matrixf_solve_chol(Matrixf* A, Matrixf* B);

// This function solves in place the linear system A*X = B for X by LTL'
// decomposition with pivoting of the square symmetric indefinite matrix A. 
// The matrix A is destroyed while B is overwritten with the matrix X. If A is 
// singular, the function returns 1. On size mismatch or non-square system,
// the function returns -1.
int matrixf_solve_ltl(Matrixf* A, Matrixf* B);

// This function solves in place the linear system A*X = B for X by LU
// decomposition with partial pivoting of the square matrix A. The matrix A
// is transformed according to LU decomposition while B is overwritten with the 
// matrix X. If A is singular, the function returns 1. On size mismatch or 
// non-square system, the function returns -1.
int matrixf_solve_lu(Matrixf* A, Matrixf* B);

// This function solves in place the linear system A*X = B for X by LU
// decomposition with partial pivoting of the banded Hessenberg matrix A. In 
// particular, A must be a Hessenberg matrix with upper bandwidth ubw >= 0. For
// instance, if ubw = 1, A is tridiagonal. Matrix A is destroyed, whereas B is 
// overwritten with the matrix X. If A is singular, the function returns 1. On
// size mismatch, non-square system or ubw < 0, the function returns -1.
int matrixf_solve_lu_banded(Matrixf* A, Matrixf* B, const int ubw);

// This function solves the linear system A*X = B by QR decomposition
// of the full-rank matrix A. The returned solution is minimum-norm. The 
// matrices A and B are destroyed. The matrix B can share the data array with X,
// provided that the array is large enough to accommodate the larger of B or X.
// If A is rank deficient, the function returns 1. On size mismatch, the 
// function returns -1.
int matrixf_solve_qr(Matrixf* A, Matrixf* B, Matrixf* X);

// This function solves the linear system A*X = B for X by QR decomposition 
// with column pivoting. The matrices A and B are destroyed. The matrix B
// can share the data array with X, provided that the array is large enough to 
// accommodate the larger of B or X. tol is the tolerance to determine the rank 
// of the A m-by-n matrix A: if the input tolerance is negative, the default 
// value max(m,n)*eps(R(0,0)) is used instead, where R(0,0) is the on-diagonal 
// element of R with the largest magnitude, being R the upper triangular matrix 
// obtained by the QR decomposition with column pivoting of A. The array work is
// the additional workspace memory: its minimum length is n. On size mismatch, 
// the function returns -1.
int matrixf_solve_qrp(Matrixf* A, Matrixf* B, Matrixf* X, float tol, float* work);

// This function solves the linear system A*X = B for X by complete orthogonal
// decomposition of the m-by-n matrix A: this ensures that X be the minimum-norm
// solution. The matrices A and B are destroyed. The matrix B can share the data
// array with X, provided that the array is large enough to accommodate the larger
// of B or X. tol is the tolerance to determine the rank of A: if the input 
// tolerance is negative, the default value max(m,n)*eps(R(0,0)) is used instead, 
// where R(0,0) is the on-diagonal element of R with the largest magnitude, being
// R the upper triangular matrix obtained by QR decomposition with column pivoting
// of A. The array work is the additional workspace memory: its minimum length is n.
// On size mismatch, the function returns -1.
int matrixf_solve_cod(Matrixf* A, Matrixf* B, Matrixf* X, float tol, float* work);

// This function substitutes the m-by-n input matrix A with its Moore-Penrose
// pseudoinverse obtained by Jacobi SVD. tol is the tolerance to treat singular
// values as zero: if the input tolerance is negative, the default value 
// max(m,n)*eps(S(0,0)) is used instead, where S(0,0) is the maximum singular 
// value of A. The array work is the additional workspace memory: its minimum 
// length is min(m,n)*(min(m,n)+1). Also, the function returns the number of 
// iterations performed.
int matrixf_pseudoinv(Matrixf* A, float tol, float* work);

// This function computes the exponential of the square matrix A by scaling
// and squaring algorithm with Padé approximation. The array work is the
// additional workspace memory: if A is n-by-n, its minimum length is n*(3*n+1).
// The function returns -1 if the input matrix is not square; it returns 1 if a
// singularity is detected.
int matrixf_exp(Matrixf* A, float* work);

// This function raises the n-by-n matrix A to the p-th positive integer power.
// The algorithm is based on the binary expansion of p to minimize the number of
// matrix multiplications. The array work is the additional workspace memory:
// its minimum length is 2*n*n. The function returns -1 if the input matrix is 
// not square.
int matrixf_pow(Matrixf* A, unsigned const int p, float* work);

// This function performs the general matrix multiplication (GEMM), which has
// the form C = alpha*op(A)*op(B) + beta*C, where A, B and C are general 
// matrices, alpha and beta are scalars, and op(X) is one of X or its
// transpose. Therefore, if transX = 0, op(X) = X, else op(X) = X', being 
// X' the transpose of X. C must be a distinct instance with respect to A and B. 
// On size mismatch, the function returns -1.
int matrixf_multiply(Matrixf* A, Matrixf* B, Matrixf* C,
	const float alpha, const float beta, const int transA, const int transB);

#endif
