#ifndef DETECTUM_H
#define DETECTUM_H

#include <math.h>

typedef struct {
	int rows; // number of rows (must be < 2^24)
	int cols; // number of columns (must be < 2^24)
	float* data; // pointer to data array
} Matrixf;

// Machine precision
#ifndef DETECTUM_EPS 
#define DETECTUM_EPS (1.1920929e-07f)
#endif

// Blue's underflow threshold
#ifndef DETECTUM_TSML
#define DETECTUM_TSML (1.0842022e-19f) 
#endif

// Blue's overflow threshold
#ifndef DETECTUM_TBIG
#define DETECTUM_TBIG (4.5035996e+15f) 
#endif

// This macro initializes a rows-by-cols matrix A, allocating its data 
// memory on the stack. rows and cols must be known at compile time.
#define Matrixf(A, rows, cols) \
float A##_data[(rows) * (cols)] = { 0 }; \
Matrixf A = { rows, cols, A##_data }

// This macro accesses the element A(i, j).
#define at(A, i, j) ((A)->data[(i) + (j) * (A)->rows])

// This function returns the positive distance from abs(x) to the next 
// larger floating-point number.
static inline float epsf(float x)
{
	return powf(2.0f, floorf(log2f(fabsf(x))) - 23.0f);
}

// This function computes 2-norm of vector v without under/overflow. 
// The length of the vector is len, and stride is its increment. 
static inline float normf(const float* v, int len, int stride)
{
	int i;
	float s = 0, h = 0, a;
	const float tsml = DETECTUM_TSML;
	const float tbig = DETECTUM_TBIG;

	for (i = 0; i < len; i++) {
		a = v[i * stride];
		if ((fabsf(a) > tsml) && (fabsf(a) < tbig)) {
			s += a * a;
		}
		else {
			h = hypotf(h, a);
		}
	}
	return hypotf(sqrtf(s), h);
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

// This function generates a Householder vector. The vector x is
// transformed so that x(0) is the norm of x and v = [1; x(1:end)],
// where v is the normalized Householder vector. len is x length 
// and stride is its increment. Also, the function returns beta
// such that H = I - beta*v*v' is a Householder matrix.
static inline float housef(float* x, int len, int stride)
{
	int i;
	float b, beta = 0;
	const float a = x[0];
	const float eps = DETECTUM_EPS;
	const float nrm = normf(x + stride, len - 1, stride);

	if (nrm > eps) {
		b = a < 0 ? hypotf(a, nrm) : -hypotf(a, nrm);
		beta = (b - a) / b;
		x[0] = b;
		for (i = 1; i < len; i++) {
			x[stride * i] /= a - b;
		}
	}
	else {
		for (i = 1; i < len; i++) {
			x[stride * i] = 0;
		}
	}
	return beta;
}

#ifdef EOF // include stdio.h before detectum.h to enable this section
// This function prints the matrix A on the standard output according to
// the specified format.
static inline void matrixf_print(Matrixf* A, const char* format)
{
	int i, j;

	for (i = 0; i < A->rows; i++) {
		for (j = 0; j < A->cols; j++) {
			printf(format, at(A, i, j));
		}
		printf("\n");
	}
}
#endif

#ifdef RAND_MAX // include stdlib.h before detectum.h to enable this section
// This function initializes a rows-by-cols matrix, allocating its data 
// memory on the heap. On allocation failure, the data pointer is null.
static inline Matrixf matrixf(int rows, int cols)
{
	Matrixf A = { rows, cols, calloc(sizeof(float), rows * cols) };

	return A;
}
#endif

// This function initializes the rows-by-cols matrix A. The array data
// collects the matrix elements, which must be stored in column-major order.
// If the data are initially arranged in row-major layout, the flag ordmem 
// can be raised to enable the conversion to column-major memory order.
void matrixf_init(Matrixf* A, int rows, int cols, float* data, int ordmem);

// This function permutes the m-by-n matrix A according to the vector of 
// permutation indices perm, which encodes the permutation matrix P: if perm is 
// m-by-1, the rows of A are permuted, whereas, if perm is 1-by-n, the columns
// of A are permuted. If the flag reverse is enabled, the multiplication order 
// between A and P is reversed, turning a column permutation into a row 
// permutation and vice versa (assuming the dimensions are compatible). If the
// flag transP is enabled, the permutation is applied according to the 
// transpose of P. The following table summarizes how A is transformed.
// ------------------------------------------------------------------------
// Is perm a row- or column-permutation vector? | reverse | transP | Output 
// ---------------------------------------------|---------|--------|-------
//                      row                     |   no    |  no    |  P*A
//                      row                     |   no    |  yes   |  P'*A
//                      row                     |   yes   |  no    |  A*P
//                      row                     |   yes   |  yes   |  A*P'
//                     column                   |   no    |  no    |  A*P
//                     column                   |   no    |  yes   |  A*P'
//                     column                   |   yes   |  no    |  P*A
//                     column                   |   yes   |  yes   |  P'*A
// ------------------------------------------------------------------------
// In general, the vector perm is transformed to encode the new permutation. 
// On size mismatch, the function returns -1. On success, it returns 0.
int matrixf_permute(Matrixf* A, Matrixf* perm, int reverse, int transP);

// This function transposes in place the input matrix.
void matrixf_transpose(Matrixf* A);

// This function performs the Cholesky decomposition of the square symmetric 
// positive definite matrix A such that A = R'*R. The upper triangular part of A 
// is overwritten by R. The function returns -1 if A is not square, -2 if A is 
// not positive definite. On success, it returns 0.
int matrixf_decomp_chol(Matrixf* A);

// This function performs the LU decomposition with partial pivoting of the 
// n-by-n matrix A so that A = P'*L*U. The matrix A is transformed so that 
// its upper triangular part stores the matrix U, whereas its strictly lower
// triangular part contains the matrix L, assuming that all the entries of 
// the main diagonal of L are ones (unit lower triangular matrix). If perm 
// is a non-null pointer, the function returns the vector of permutations 
// that encodes the permutation matrix P. The vector perm must be initialized
// as a n-by-1 vector. Additionally, one can provide the matrix B, which is 
// transformed into P*B; this computation is skipped if B is a null pointer. 
// On size mismatch or non-square matrix, the function returns -1. On success,
// it returns 0.
int matrixf_decomp_lu(Matrixf* A, Matrixf* perm, Matrixf* B);

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
// transformed into matrix R. If Q is a null pointer, the explicit formation of 
// the orthogonal matrix is omitted, and the strictly lower triangular part of A 
// stores the essential parts of the Householder vectors, which can be used to
// accumulate Q afterwards. For the sake of clarity, the essential part of a 
// Householder vector comprises all elements but the first, which is always 1. 
// If perm is a non-null pointer, the decomposition makes use of column pivoting 
// so that A*P = Q*R, where P is the permutation matrix encoded by the vector
// perm. The vector perm must be initialized as a 1-by-n vector. Additionally, 
// one can provide the matrix B, which is transformed into Q'*B; this computation
// is skipped if B is a null pointer. If m > n, the function can also produce the
// economy-size decomposition such that only the first n columns of Q are 
// computed (thin Q) and the last m - n rows of R are excluded so that R becomes 
// n-by-n. To enable the economy-size decomposition, Q must be initialized as an
// m-by-n matrix. On size mismatch, the function returns -1. On success, it 
// returns 0.
int matrixf_decomp_qr(Matrixf* A, Matrixf* Q, Matrixf* perm, Matrixf* B);

// This function unpacks an orthogonal matrix from its factored representation. 
// In particular, if flag fwd > 0, the function transforms B into Q'*B by forward
// accumulation of Householder matrices. Otherwise, if fwd <= 0, the function 
// transforms B into Q*B by backward accumulation of Householder matrices. Matrix
// A remains the same. The lower triangular part of A below the s-th subdiagonal
// must store the essential parts of the Householder vectors. For instance, if 
// s = 0, the Householder vectors are below the main diagonal of A; whereas, if 
// s = 1, the Householder vectors are below the first subdiagonal of A. On size
// mismatch or if s < 0, the function returns -1. On success, it returns 0.
int matrixf_unpack_house(Matrixf* A, Matrixf* B, int s, int fwd);

// This function accomplishes the bidiagonalization of the m-by-n matrix A so
// that A = U*B*V'. A is overwritten by the bidiagonal matrix B. Specifically,
// B is upper bidiagonal if m >= n, otherwise B is lower bidiagonal. The
// explicit formation of the matrices U and V is omitted if these are null 
// pointers. If U (V) is a null pointer, the essential parts of the Householder
// vectors are stored off the bidiagonal band and can be used to accumulate U 
// (V) afterwards. In particular: 
// - if m >= n, U is encoded below the main diagonal of A, while V is encoded 
//   below the first subdiagonal of A';
// - if m < n, U is encoded below the first subdiagonal of A, while V is encoded 
//   below the main diagonal of A'.
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
// whose size is r-by-r, where r is the rank of A. The permutation vector perm, 
// which encodes the permutation matrix P, must be always provided, as the function 
// makes use of QR decomposition with column pivoting. The vector perm must be 
// initialized as a 1-by-n vector. If U is a null pointer, the computation of the
// matrix U is omitted. If V is a null pointer, the explicit formation of the matrix 
// V is omitted; in this case, the essential parts of the Householder vectors are 
// stored below the main diagonal of L' and can be used to accumulate V afterwards. 
// If m > n, the function can also produce the economy-size decomposition such that 
// only the first n columns of U are computed (thin U) and the last m - n rows of T
// are excluded so that T becomes n-by-n. To enable the economy-size decomposition, 
// U must be initialized as an m-by-n matrix. tol is the tolerance to determine the 
// rank of A: if the input tolerance is negative, the default value 
// max(m,n)*eps(R(0,0)) is used instead, where R(0,0) is the on-diagonal element of 
// R with the largest magnitude, R being the upper triangular matrix obtained by QR
// decomposition with column pivoting of A. On size mismatch, the function returns 
// -1. On success, it returns the rank of A.
int matrixf_decomp_cod(Matrixf* A, Matrixf* U, Matrixf* V, Matrixf* perm, float tol);

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
// of iterations is reached: results may be inaccurate. On success, it returns the
// number of iterations performed. 
// Related macros
// - DETECTUM_SVD_ITER_MAX: maximum number of iterations. The default value is 
//   100*min(m,n).  
// - DETECTUM_SVD_TOL: tolerance. The default value is 1e-6.
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
// The function returns -1 on size mismatch. It returns -2 if the maximum 
// number of iterations is reached: results may be inaccurate. On success, it
// returns the number of iterations performed.
// Related macros
// - DETECTUM_SVD_JACOBI_ITER_MAX: maximum number of iterations. The default
//   value is 10*min(m,n).
// - DETECTUM_SVD_JACOBI_TOL: tolerance. The default value is 1e-6.
int matrixf_decomp_svd_jacobi(Matrixf* A, Matrixf* U, Matrixf* V);

// This function performs the Hessenberg decomposition of the square matrix A, 
// which is transformed into the Hessenberg matrix H so that A = P*H*P'. If P 
// is a null pointer, the explicit formation of the orthogonal matrix is omitted,
// and the triangular part of A below the first subdiagonal stores the essential 
// parts of the Householder vectors, which can be used to accumulate P afterwards. 
// The function returns -1 if the matrices are not square or on size mismatch.
// On success, it returns 0.
int matrixf_decomp_hess(Matrixf* A, Matrixf* P);

// This function performs the Schur decomposition of the symmetric matrix A so that
// A = U*D*U'. A is transformed into the diagonal matrix D whose main diagonal holds
// the eigenvalues of A. If U is a null pointer, the computation of the orthogonal 
// matrix is omitted. The function returns -1 if the matrices are not square or on 
// size mismatch. It returns -2 if the maximum number of iterations is reached: 
// results may be inaccurate. On success, it returns the number of iterations 
// performed.
// Related macros
// - DETECTUM_SCHUR_SYMM_ITER_MAX: maximum number of iterations. The default value
//   is 100*n, where n is the number of rows.
// - DETECTUM_SCHUR_SYMM_TOL: tolerance. The default value is 1e-6.
int matrixf_decomp_schur_symm(Matrixf* A, Matrixf* U);

// This function performs the Schur decomposition of the square matrix A so that
// A = U*T*U'. A is transformed into the quasitriangular matrix T whose main diagonal
// holds the real eigenvalues, whereas the complex eigenvalues are expressed as
// 2-by-2 blocks along the main diagonal. In particular, each 2-by-2 block has the
// form B = [x, y; z, x], and its complex conjugate eigenvalues, i.e. x+sqrt(-y*z)*i 
// and x-sqrt(-y*z)*i are also a complex conjugate pair of eigenvalues of A. 
// If U is a null pointer, the computation of the orthogonal matrix is omitted. The 
// function returns -1 if the matrices are not square or on size mismatch. It returns
// -2 if the maximum number of iterations is reached: results may be inaccurate. On 
// success, it returns the number of iterations performed.
// Related macros
// - DETECTUM_SCHUR_ITER_MAX: maximum number of iterations. The default value is 
//   100*n, where n is the number of rows.
// - DETECTUM_SCHUR_TOL: tolerance. The default value is 1e-6.
// - DETECTUM_SCHUR_AD_HOC_SHIFT_COUNT: iteration count defining the period for the
//   application of ad hoc shifts. The default value is 5.
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
// minimum length is max(2,4*(n-2)^2); if all the eigenvalues are real, the 
// minimum length of work is max(2,(n-1)^2). If the matrices of eigenvectors are
// not n-by-n, the function returns -1; if an eigenvector cannot be computed due to
// a singularity, it returns -2. On success, it returns 0.
int matrixf_get_eigenvectors(Matrixf* T, Matrixf* U,
	Matrixf* V, Matrixf* W, int pseudo, float* work);

// This function performs the forward substitution on the lower-triangular
// matrix L to solve the the system L*X = B for X. The flag unitri indicates
// whether L is unitriangular: if the flag is enabled, the elements on the main
// diagonal are ignored. Matrix L remains the same while B is destroyed. Also, 
// the matrix B can share the data array with X, provided that the array is 
// large enough to accommodate the larger of B or X. If L is rank deficient, 
// it returns -2. On size mismatch, it returns -1. On success, it returns 0.
int matrixf_solve_tril(Matrixf* L, Matrixf* B, Matrixf* X, int unitri);

// This function performs the backward substitution on the upper-triangular
// matrix U to solve the the system U*X = B for X. The flag unitri indicates
// whether U is unitriangular: if the flag is enabled, the elements on the main
// diagonal are ignored. Matrix U remains the same while B is destroyed. Also,
// the matrix B can share the data array with X, provided that the array is 
// large enough to accommodate the larger of B or X. If U is rank deficient, 
// it returns -2. On size mismatch, it returns -1. On success, it returns 0.
int matrixf_solve_triu(Matrixf* U, Matrixf* B, Matrixf* X, int unitri);

// This function solves in place the linear system A*X = B for X by Cholesky
// decomposition of the square symmetric positive definite matrix A. The upper 
// triangular part of A is transformed according to Cholesky decomposition
// while B is overwritten with the matrix X. If A is not positive definite, the 
// function returns -2. On size mismatch or non-square system, the function 
// returns -1. On success, it returns 0.
int matrixf_solve_chol(Matrixf* A, Matrixf* B);

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

// This function solves the linear system A*X = B by QR decomposition of the 
// full-rank matrix A. If the system is underdetermined, the returned solution 
// is of minimum norm. The matrices A and B are destroyed. The matrix B can 
// share the data array with X, provided that the array is large enough to 
// accommodate the larger of B or X. If A is rank deficient, the function 
// returns -2. On size mismatch, the function returns -1. On success, it 
// returns 0.
int matrixf_solve_qr(Matrixf* A, Matrixf* B, Matrixf* X);

// This function solves the linear system A*X = B for X by QR decomposition 
// with column pivoting of the m-by-n matrix A. The matrices A and B are 
// destroyed. The matrix B can share the data array with X, provided that the 
// array is large enough to accommodate the larger of B or X. tol is the 
// tolerance to determine the rank of A: if the input tolerance is negative,
// the default value max(m,n)*eps(R(0,0)) is used instead, where R(0,0) is the 
// on-diagonal element of R with the largest magnitude, R being the upper 
// triangular matrix obtained by the QR decomposition with column pivoting of A.
// The array work is the additional workspace memory: its minimum length is n. 
// On size mismatch, the function returns -1. On success, it returns 0.
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
// length is min(m,n)*(min(m,n)+1). The function returns -2 if the maximum number
// of iterations is reached: results may be inaccurate. On success, it returns 
// the number of iterations performed.
int matrixf_pseudoinv(Matrixf* A, float tol, float* work);

// This function computes the exponential of the square matrix A by scaling
// and squaring algorithm with Padé approximation. The array work is the
// additional workspace memory: if A is n-by-n, its minimum length is 3*n*n+n.
// The function returns -1 if the input matrix is not square; if the matrix 
// exponential cannot be computed due to a singularity, it returns -2. On 
// success, it returns 0.
// Related macros
// - DETECTUM_EXP_PADE_ORDER: order of diagonal Padé approximation. The 
//   default value is 4.
int matrixf_exp(Matrixf* A, float* work);

// This function computes the principal logarithm of matrix A by inverse scaling
// and squaring and Gregory series expansion. The n-by-n matrix A is transformed
// into its logarithm. The array work is the additional workspace memory: its 
// minimum length is 3*n*n+n. If A is not square, the function returns -1. If 
// Schur decomposition fails to converge, the function returns -2. If A has 
// nonpositive real eigenvalues, the principal logarithm does not exist, and the 
// function returns -3. If the matrix logarithm cannot be computed due to a 
// singularity, it returns -4. On success, it returns 0.
// Related macros
// - DETECTUM_LOG_ISS_THR: inverse scaling and squaring threshold. The default 
//   value is 0.5.
// - DETECTUM_LOG_NTERMS: number of terms of Gregory series expansion. The 
//   default value is 5.
int matrixf_log(Matrixf* A, float* work);

// This function computes the principal square root of the quasitriangular matrix 
// T by real Schur method. T is obtained by Schur decomposition. The n-by-n
// matrix T is transformed into X so that T = X*X. If T has negative real 
// eigenvalues, the principal square root does not exist, and the function returns
// -3. If the matrix logarithm cannot be computed due to a singularity, it returns
// -4. If T is singular, the function attempts to compute a matrix square root and,
// if no error occurs, returns 1; however, a singular matrix may not have a square
// root. Otherwise, on success, it returns 0.
int matrixf_sqrt_quasitriu(Matrixf* T);

// This function computes the principal square root of matrix A by real Schur method.
// The n-by-n matrix A is transformed into X so that A = X*X. The array work is the
// additional workspace memory: its minimum length is n*n+n. If A is not square, the
// function returns -1. If Schur decomposition fails to converge, the function 
// returns -2. If A has negative real eigenvalues, the principal square root does not
// exist, and the function returns -3. If the matrix square root cannot be computed
// due to a singularity, it returns -4. If A is singular, the function attempts to
// compute a matrix square root and, if no error occurs, returns 1; however, a 
// singular matrix may not have a square root. Otherwise, on success, it returns 0.
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

// This function sequentially performs the matrix multiplications A = op(L)*A
// and A = A*op(R), where A is m-by-n, L and R are square, and op(X) is one of
// X or its transpose. Therefore, if transX = 0, op(X) = X, else op(X) = X', X' 
// being the transpose of X. A must be a distinct instance with respect to L 
// and R. Matrices L and R remain the same, whereas A in transformed in place.
// If L (R) is a null pointer, the left (right) multiplication is omitted. The
// array work is the additional workspace memory: if only L is provided, its
// minimum length is m; if only R is provided, its minimum length is n; if both
// are provided, its minimum length is max(m,n). On size mismatch, the function
// returns -1. On success, it returns 0.
int matrixf_multiply_inplace(Matrixf* A, Matrixf* L, Matrixf* R,
	int transL, int transR, float* work);

#endif