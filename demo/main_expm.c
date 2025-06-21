#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

#define MATRIX 2
#define METHOD 1

int main()
{
#if MATRIX == 0
#define N 3
	float A_data[N * N] = {
		1, 1, 0,
		0, 0, 2,
		0, 0, -1
	};
#elif MATRIX == 1 // defective matrix
#define N 3
	float A_data[N * N] = {
		3, 1, 0,
		0, 3, 1,
		0, 0, 3
	};
#else // defective matrix
#define N 5
	float A_data[N * N] = {
		1, 1, 0, 0, 0,
		0, 1, 1, 0, 0,
		0, 0, 1, 1, 0,
		0, 0, 0, 1, 1,
		0, 0, 0, 0, 1
	};
#endif
	Matrixf A = { 0 };

	matrixf_init(&A, N, N, A_data, 1);
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
#if METHOD // using dedicated function (more accurate)
	float work[N * (3 * N + 1)] = { 0 };

	matrixf_exp(&A, work);
#else // general method based on diagonalization with perturbation
	int i, j;
	Matrixf U = matrixf(N, N);
	Matrixf V = matrixf(N, N);
	float work[4 * N * N] = { 0 };
	float normA = normf(A_data, N * N, 1);
	float scale = 100 * N * epsf(normA);
	float re, im, mag;

	// Perturb matrix A so that it becomes diagonalizable, if it is not already.
	for (i = 0; i < N * N; i++) {
		A.data[i] += scale * ((float)rand() / (float)RAND_MAX);
	}
	// Apply Schur decomposition so that A = U*T*U'
	matrixf_decomp_schur(&A, &U);
	// Get right pseudo-eigenvectors so that A = V*D*inv(V)
	matrixf_get_eigenvectors(&A, &U, &V, 0, 1, work);
	// Transform T into exp(D)
	for (j = 0; j < N; j++) {
		if (j == N - 1 || at(&A, j + 1, j) == 0) { // real eigenvalue
			for (i = 0; i < j; i++) {
				at(&A, i, j) = 0;
			}
			at(&A, j, j) = expf(at(&A, j, j));
		}
		else { // 2-by-2 block with complex conjugate pair of eigenvalues
			for (i = 0; i < j; i++) {
				at(&A, i, j) = at(&A, i, j + 1) = 0;
			}
			re = at(&A, j, j);
			im = sqrtf(-at(&A, j + 1, j) * at(&A, j, j + 1));
			mag = expf(re);
			at(&A, j, j) = mag * cosf(im);
			at(&A, j + 1, j) = -mag * sinf(im);
			at(&A, j + 1, j + 1) = at(&A, j, j);
			at(&A, j, j + 1) = -at(&A, j + 1, j);
			j++;
		}
	}
	// Get exp(A) as V*exp(D)*inv(V)
	matrixf_multiply(&V, &A, &U, 1, 0, 0, 0);
	matrixf_transpose(&U);
	matrixf_transpose(&V);
	if (matrixf_solve_lu(&V, &U)) {
		return -1;
	}
	matrixf_transpose(&U);
	for (i = 0; i < N * N; i++) {
		A.data[i] = U.data[i];
	}
	free(U.data);
	free(V.data);
#endif
	printf("exp(A) = \n"); matrixf_print(&A, "%9.4f "); printf("\n");

	return 0;
}