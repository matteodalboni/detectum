#include "detego.h"
#include "detego_utils.h"

#define ALGO 2
#define TEST 0
#define ITER 50

static void copy_matrix(Matrixf* Dest, Matrixf* Src)
{
	int len = Src->size[0] * Src->size[1];

	Dest->size[0] = Src->size[0];
	Dest->size[1] = Src->size[1];
	while (len--) Dest->data[len] = Src->data[len];
}

int main()
{
#if (TEST == 0)
#define m 8
#define n 8
	float A_data[] = {
		 0, 1, 0, 0, 0, 0, 0, 1,
		 1, 0, 1, 0, 0, 0, 0, 0,
		 0, 1, 0, 1, 0, 0, 0, 0,
		 0, 0, 1, 0, 1, 0, 0, 0,
		 0, 0, 0, 1, 0, 1, 0, 0,
		 0, 0, 0, 0, 1, 0, 1, 0,
		 0, 0, 0, 0, 0, 1, 0, 1,
		-1, 0, 0, 0, 0, 0, 1, 0
	};
#elif (TEST == 1)
#define m 4
#define n 5
	float A_data[] = {
		17, 24,  1,  8, 15,
		23,  5,  7, 14, 16,
		 4,  6, 13, 20, 22,
		10, 12, 19, 21,  3
	};
#elif (TEST == 2)
#define m 5
#define n 4
	float A_data[m * n] = {
		17, 24,  1,  8,
		23,  5,  7, 14,
		4,  6, 13, 20,
		10, 12, 19, 21,
		11, 18, 25,  2
	};
#elif (TEST == 3)
#define m 6
#define n 5
	float A_data[m * n] = {
		0,  1, 0, 26, 26,
		0, 32, 0, 21, 21,
		0,  9, 0, 22, 22,
		0, 28, 0, 17, 17,
		0,  5, 0, 12, 12,
		0, 36, 0, 13, 13
	};
#else
#define m 6
#define n 5
	float A_data[m * n] = { 0 };
#endif
#define p (m > n ? m : n)

	int i, j, iter = ITER;
	float s;
	float A_copy[m * n] = { 0 };
	float U_data[m * m] = { 0 };
	float V_data[n * n] = { 0 };
	float P_data[p * p] = { 0 };
	float Q_data[p * p] = { 0 };
	float W_data[p * p] = { 0 };
	Matrixf A = { 0 };
	Matrixf U = { { m, m }, U_data };
	Matrixf V = { { n, n }, V_data };
	Matrixf P = { { m, m }, P_data };
	Matrixf Q = { { m, m }, Q_data };
	Matrixf W = { { m, n }, W_data };
	Matrixf* X;

	matrixf_init(&A, m, n, A_data, 1);
	for (i = 0; i < m * n; i++) A_copy[i] = A_data[i];
	printf("A = \n"); PRINT("%9.4f ", &A);
#if (ALGO == 0)
	for (i = 0; i < p * p; i++) {
		if (i < m * m) P.data[i] = !(i % (m + 1));
		if (i < m * m) U.data[i] = !(i % (m + 1));
		if (i < n * n) V.data[i] = !(i % (n + 1));
	}

	for (j = 0; j < 2 * iter; j++) {
		copy_matrix(&W, &A);
		matrixf_multiply(&P, &W, &A, 1, 0, 0, 0);
		P.size[0] = A.size[1]; P.size[1] = A.size[1];
		Q.size[0] = A.size[0]; Q.size[1] = A.size[0];
		matrixf_decomp_qr(&A, &Q, &P, 0);
		X = (j % 2) ? &V : &U;
		copy_matrix(&W, X);
		matrixf_multiply(&W, &Q, X, 1, 0, 0, 0);
		matrixf_transpose(&A);
	}

	copy_matrix(&W, &U);
	matrixf_multiply(&W, &P, &U, 1, 0, 0, 0);
	for (j = 0; j < (m < n ? m : n); j++) {
		s = at(&A, j, j) < 0 ? -1.0f : 1.0f;
		for (i = 0; i < n; i++) at(&V, i, j) *= s;
		for (i = 0; i < m; i++) at(&A, i, j) *= s;
	}
#elif (ALGO == 1)
	matrixf_init(&Q, n, n, Q_data, 0);
	matrixf_init(&P, 1, n, P_data, 0);
	matrixf_multiply(&A, &A, &Q, 1, 0, 1, 0);
	matrixf_decomp_schur_symm(&Q, &V);
	matrixf_multiply(&A, &V, &W, 1, 0, 0, 0);
	matrixf_decomp_qr(&W, &U, &P, 0);
	matrixf_permute(&V, &P, 0);
	for (j = 0; j < (m < n ? m : n); j++) {
		if (at(&W, j, j) < 0) {
			at(&W, j, j) *= -1.0f;
			for (i = 0; i < m; i++)
				at(&U, i, j) *= -1.0f;
		}
		for (i = 0; i < j; i++) 
			at(&W, i, j) = 0;
	}
	copy_matrix(&A, &W);
#elif (ALGO == 2)
	if (matrixf_decomp_svd_jacobi(&A, &U, &V) < 0) return -1;
#else
	if (matrixf_decomp_svd(&A, &U, &V) < 0) return -1;
#endif
	printf("U = \n"); PRINT("%9.4f ", &U);
	printf("S = \n"); PRINT("%9.4f ", &A);
	printf("V = \n"); PRINT("%9.4f ", &V);

	copy_matrix(&W, &A);
	matrixf_multiply(&U, &A, &W, 1, 0, 0, 0);

	matrixf_multiply(&W, &V, &A, 1, 0, 0, 1);
	printf("U*S*V' = \n"); PRINT("%9.4f ", &A);
	for (i = 0; i < m * n; i++) A_copy[i] -= A.data[i];
	printf("||A - U*S*V'||_F = %g\n\n", get_norm2(A_copy, m * n, 1));

	matrixf_init(&P, m, m, P_data, 0);
	matrixf_multiply(&U, &U, &P, 1, 0, 0, 1); 
	printf("U*U' = \n"); PRINT("%9.4f ", &P);
	matrixf_multiply(&U, &U, &P, 1, 0, 1, 0);
	printf("U'*U = \n"); PRINT("%9.4f ", &P);

	matrixf_init(&Q, n, n, Q_data, 0);
	matrixf_multiply(&V, &V, &Q, 1, 0, 0, 1);
	printf("V*V' = \n"); PRINT("%9.4f ", &Q);
	matrixf_multiply(&V, &V, &Q, 1, 0, 1, 0);
	printf("V'*V = \n"); PRINT("%9.4f ", &Q);

	return 0;
}