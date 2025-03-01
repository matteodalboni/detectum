#include "detego.h"
#include "detego_utils.h"

// This function extracts the eigenvalues from the quasitriangular Schur 
// matrix T. The real and imaginary parts are stored in lambda_re and lambda_im, 
// respectively.
static void get_eigvals(Matrixf* T, float* lambda_re, float* lambda_im)
{
	int i = 0;
	const int n = T->size[0];
	float t, s, r;

	while (i < n)
	{
		if (i == n - 1 || _(T, i + 1, i) == 0) {
			lambda_re[i] = _(T, i, i);
			lambda_im[i] = 0;
			i += 1;
		}
		else {
			t = 0.5f * (_(T, i, i) - _(T, i + 1, i + 1));
			s = t * t + _(T, i + 1, i) * _(T, i, i + 1);
			r = (s < 0) ? sqrtf(-s) : sqrtf(s);
			lambda_re[i] = lambda_re[i + 1] = _(T, i + 1, i + 1) + t;
			lambda_im[i] = lambda_im[i + 1] = 0;
			if (s >= 0) {
				lambda_re[i + 0] -= r;
				lambda_re[i + 1] += r;
			}
			else {
				lambda_im[i + 0] -= r;
				lambda_im[i + 1] += r;
			}
			i += 2;
		}
	}
}

static void get_eigvals2x2(float* lambda,
	float a00, float a01, float a10, float a11)
{
	float t = 0.5f * (a00 - a11);
	float s = t * t + a10 * a01;
	float r = (s < 0) ? sqrtf(-s) : sqrtf(s);

	lambda[0] = lambda[2] = a11 + t;
	lambda[1] = lambda[3] = 0;
	lambda[0 + (s < 0)] -= r;
	lambda[2 + (s < 0)] += r;
}

static int is_quasitriu(Matrixf* T, float tol)
{
	int i = 0;
	const int n = T->size[0];
	const float eps = tol * get_norm2(T->data, n * n, 1);

	while (i < n - 2) {
		if (fabsf(_(T, i + 1, i)) <= eps) {
			_(T, i + 1, i) = 0; i += 1;
		}
		else {
			if (fabsf(_(T, i + 2, i + 1)) <= eps) {
				_(T, i + 2, i + 1) = 0; i += 2;
			}
			else return 0;
		}
	}

	return 1;
}

#define ALGO 2
#define TEST 5
#define ITER 1000
#define TOL 1e-6f

float A_data[] = {
#if (TEST == 0) // gallery(3)
	-149, -50, -154,
	 537, 180,  546,
	 -27,  -9,  -25
#elif (TEST == 1) // symmetric with repeated eigenvalues
	3, 2, 4,
	2, 0, 2,
	4, 2, 3
#elif (TEST == 2) // nonsymmetric with repeated real eigenvalues
	 6,  12,  19,
	-9, -20, -33,
	 4,  9,   15
#elif (TEST == 3) // gallery('minij',10)
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	1, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	1, 2, 3, 4, 4, 4, 4, 4, 4, 4,
	1, 2, 3, 4, 5, 5, 5, 5, 5, 5,
	1, 2, 3, 4, 5, 6, 6, 6, 6, 6,
	1, 2, 3, 4, 5, 6, 7, 7, 7, 7,
	1, 2, 3, 4, 5, 6, 7, 8, 8, 8,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 9,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#elif (TEST == 4) // A(n,n) = 0 and complex conjugate pair of eigenvalues
	0, 0, 1,
	1, 0, 0,
	0, 1, 0
#elif (TEST == 5) // complex conjugate pair of eigenvalues 
	 1, 1, 1, 3,
	 1, 2, 1, 1,
	 1, 1, 3, 1,
	-2, 1, 1, 4
#elif (TEST == 6) // gallery('hanowa',10)
	-1,  0,  0,  0,  0, -1,  0,  0,  0,  0,
	 0, -1,  0,  0,  0,  0, -2,  0,  0,  0,
	 0,  0, -1,  0,  0,  0,  0, -3,  0,  0,
	 0,  0,  0, -1,  0,  0,  0,  0, -4,  0,
	 0,  0,  0,  0, -1,  0,  0,  0,  0, -5,
	 1,  0,  0,  0,  0, -1,  0,  0,  0,  0,
	 0,  2,  0,  0,  0,  0, -1,  0,  0,  0,
	 0,  0,  3,  0,  0,  0,  0, -1,  0,  0,
	 0,  0,  0,  4,  0,  0,  0,  0, -1,  0,
	 0,  0,  0,  0,  5,  0,  0,  0,  0, -1
#elif (TEST == 7)
	0, 0, 0, -1e5f,
	0, 0, -1e5f, 0,
	0, -1e5f, 0, 0,
	-1e5f, 0, 0, 0
#elif (TEST == 8) // see "Variants of the QR Algorithm" By Cleve Moler
	0, 2, 0, -1,
	1, 0, 0,  0,
	0, 1, 0,  0,
	0, 0, 1,  0
#elif (TEST == 9) // see "Variants of the QR Algorithm" By Cleve Moler
	0,     1,      0, 0,
	1,     0, -1e-4f, 0,
	0, 1e-4f,      0, 1,
	0,     0,      1, 0
#elif (TEST == 10) // permutation matrix
	0, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 1, 0, 0, 0,
	0, 1, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 1, 0, 0,
	0, 0, 0, 1, 0, 0, 0, 0,
	0, 0, 1, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 0
#elif (TEST == 11)
	0, 1,
	1, 0
#endif
};
#define NUMEL (sizeof(A_data) / sizeof(A_data[0]))

#if (ALGO == 0) 
// Single shift QR iteration
int main()
{
	int i, n = (int)sqrtf(NUMEL), iter = ITER;
	float s = 0;
	const float eps = EPS(1);
	float lambda[NUMEL] = { 0 };
	float U_data[NUMEL] = { 0 };
	float Q_data[NUMEL] = { 0 };
	float R_data[NUMEL] = { 0 };
	Matrixf A = { {n, n}, A_data };
	Matrixf U = { {n, n}, U_data };
	Matrixf Q = { {n, n}, Q_data };
	Matrixf R = { {n, n}, R_data };

	matrixf_transpose(&A);
	printf("A = \n"); PRINT("%9.4f ", &A);

	matrixf_decomp_hess(&A, &U);
	matrixf_transpose(&U);
	while (iter && !is_quasitriu(&A, TOL)) {
		s = _(&A, n - 1, n - 1);
#if 1 // Demmel shift
		get_eigvals2x2(lambda,
			_(&A, n - 2, n - 2), _(&A, n - 2, n - 1),
			_(&A, n - 1, n - 2), _(&A, n - 1, n - 1));
		s = fabsf(lambda[0] - s) < fabsf(lambda[2] - s) ? lambda[0] : lambda[2];
#endif
		if (s == 0) s = eps;
		for (i = 0; i < n; i++) _(&A, i, i) -= s;
		matrixf_decomp_qr(&A, &Q, 0, &U);
		for (i = 0; i < n * n; i++) R.data[i] = A.data[i];
		matrixf_multiply(&R, &Q, &A, 1, 0, 0, 0);
		for (i = 0; i < n; i++) _(&A, i, i) += s;
		iter--;
	}
	matrixf_transpose(&U);

	printf("# iterations: %d\n\n", ITER - iter);

	for (i = 0; i < n * n; i++) R.data[i] = A.data[i];
	printf("T = \n"); PRINT("%9.4f ", &R);
	printf("U = \n"); PRINT("%9.4f ", &U);
	get_eigvals(&R, lambda, lambda + n);

	matrixf_multiply(&U, &R, &Q, 1, 0, 0, 0);
	matrixf_multiply(&Q, &U, &A, 1, 0, 0, 1);
	printf("U*T*U' = \n"); PRINT("%9.4f ", &A);

	printf("Eigenvalues\n");
	for (i = 0; i < n; i++) {
		printf("%4d: %+.4f", i + 1, lambda[i]);
		if (lambda[i + n]) printf("%+.4fi", lambda[i + n]);
		printf("\n");
	}

	return 0;
}
#elif (ALGO == 1)
// Orthogonal iteration (slow to converge with nonsymmetric matrices
// when the magnitudes of some eigenvalues are close to each other)
int main()
{
	int i, n = (int)sqrtf(NUMEL), iter = ITER;
	float lambda[NUMEL] = { 0 };
	float P_data[NUMEL] = { 0 };
	float U_data[NUMEL] = { 0 };
	float T_data[NUMEL] = { 0 };
	Matrixf A = { {n, n}, A_data };
	Matrixf P = { {n, n}, P_data };
	Matrixf U = { {n, n}, U_data };
	Matrixf T = { {n, n}, T_data };

	matrixf_transpose(&A);
	printf("A = \n"); PRINT("%9.4f ", &A);

	matrixf_decomp_hess(&A, &P);
	printf("H = \n"); PRINT("%9.4f ", &A);
	printf("P = \n"); PRINT("%9.4f ", &P);

	//matrixf_multiply(&P, &A, &T, 1, 0, 0, 0);
	//matrixf_multiply(&T, &P, &A, 1, 0, 0, 1);
	//printf("P*H*P' = \n"); PRINT("%9.4f ", &A);

	for (i = 0; i < n * n; i++) T.data[i] = A.data[i];

	while (iter--) {
		matrixf_decomp_qr(&T, &U, 0, 0);
		matrixf_multiply(&A, &U, &T, 1, 0, 0, 0);
	}

	for (i = 0; i < n * n; i++) A.data[i] = T.data[i];
	matrixf_multiply(&U, &A, &T, 1, 0, 1, 0);
	for (i = 0; i < n * n; i++) A.data[i] = U.data[i];
	matrixf_multiply(&P, &A, &U, 1, 0, 0, 0);

	for (i = 0; i < n - 1; i++)
		if (fabsf(_(&T, i + 1, i)) < TOL) _(&T, i + 1, i) = 0;

	printf("T = \n"); PRINT("%9.4f ", &T);
	printf("U = \n"); PRINT("%9.4f ", &U);
	get_eigvals(&T, lambda, lambda + n);

	matrixf_multiply(&U, &T, &A, 1, 0, 0, 0);
	matrixf_multiply(&A, &U, &T, 1, 0, 0, 1);
	printf("U*T*U' = \n"); PRINT("%9.4f ", &T);

	printf("Eigenvalues\n");
	for (i = 0; i < n; i++) {
		printf("%4d: %+.4f", i + 1, lambda[i]);
		if (lambda[i + n]) printf("%+.4fi", lambda[i + n]);
		printf("\n");
	}

	return 0;
}
#else 
// Francis double implicit shift QR iteration
int main()
{
	int i, n = (int)sqrtf(NUMEL), iter = ITER;
	float lambda[NUMEL] = { 0 };
	float U_data[NUMEL] = { 0 };
	float B_data[NUMEL] = { 0 };
	Matrixf A = { {n, n}, A_data };
	Matrixf U = { {n, n}, U_data };
	Matrixf B = { {n, n}, B_data };

	matrixf_transpose(&A);
	DISP("%9.4f ", A);

	iter = matrixf_decomp_schur(&A, &U);

	printf("# iterations: %d\n\n", iter);

	printf("T = \n"); PRINT("%9.4f ", &A);
	printf("U = \n"); PRINT("%9.4f ", &U);
	get_eigvals(&A, lambda, lambda + n);

	matrixf_multiply(&U, &A, &B, 1, 0, 0, 0);
	matrixf_multiply(&B, &U, &A, 1, 0, 0, 1);
	printf("U*T*U' = \n"); PRINT("%9.4f ", &A);

	printf("Eigenvalues\n");
	for (i = 0; i < n; i++) {
		printf("%4d: %+.5f", i + 1, lambda[i]);
		if (lambda[i + n]) printf("%+.5fi", lambda[i + n]);
		printf("\n");
	}

	return 0;
}
#endif