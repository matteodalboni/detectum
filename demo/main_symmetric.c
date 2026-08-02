#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

#define N (7)
#define SEED (2002)
//#define POSITIVE_DEFINITE

static void make_positive_definite(Matrixf* A)
{
	Matrixf(B, N, N);

	for (int i = 0; i < N * N; i++) {
		B.data[i] = A->data[i];
	}
	matrixf_multiply(&B, &B, A, 1.0f, 0.0f, 0, 1);
}

int main()
{
	int i, j, positive_definite;
	float work[N], copy[N * N];
	Matrixf(A, N, N);
	Matrixf(T, N, N);
	Matrixf perm = { N, 1, A.data };
	Matrixf Lred = { N, N - 1, A.data + N };

	srand(SEED);
	for (j = 0; j < N; j++) {
		for (i = j; i < N; i++) {
			at(&A, i, j) = at(&A, j, i) =
				2.0f * ((float)rand() / (float)RAND_MAX - 0.5f);
		}
	}
#ifdef POSITIVE_DEFINITE
	make_positive_definite(&A);
#endif
	for (i = 0; i < N * N; i++) {
		copy[i] = A.data[i];
	}
	printf("A = \n"); matrixf_print(&A, "%9.4f");

	printf("\nEigendecomposition\n");
	matrixf_decomp_schur_symm(&A, &T);
	printf("D = \n"); matrixf_print(&A, "%9.4f");
	printf("U = \n"); matrixf_print(&T, "%9.4f");
	i = 0;
	while (at(&A, i, i) > 0.0f && i < N) {
		i++;
	}
	positive_definite = i == N ? 1 : 0;
	matrixf_multiply_inplace(&A, &T, &T, 0, 1, work);
	printf("U*D*U' = \n"); matrixf_print(&A, "%9.4f");
	for (i = 0; i < N * N; i++) {
		T.data[i] = 0.0f;
	}

	if (positive_definite) {
		printf("\nCholesky decomposition\n");
		matrixf_decomp_chol(&A);
		for (j = 0; j < N; j++) {
			for (i = j + 1; i < N; i++) {
				at(&A, i, j) = 0.0f;
			}
		}
		printf("R = \n"); matrixf_print(&A, "%9.4f");
		matrixf_multiply(&A, &A, &T, 1.0f, 0.0f, 1, 0);
		printf("R'*R = \n"); matrixf_print(&T, "%9.4f");
	}
	else {
		printf("\nAasen decomposition\n");
		matrixf_decomp_ltl(&A);
		for (j = 0; j < N - 1; j++) {
			at(&T, j, j) = at(&A, j, j);
			at(&A, j, j) = 1.0f;
			at(&T, j, j + 1) = at(&T, j + 1, j) = at(&A, j, j + 1);
			at(&A, j, j + 1) = 0.0f;
		}
		at(&T, N - 1, N - 1) = at(&A, N - 1, N - 1);
		at(&A, N - 1, N - 1) = 1.0f;
		perm.data[0] = 0.0f;
		matrixf_permute(&Lred, &perm, 0, 1);
		perm.data[0] = 1.0f;
		at(&A, 0, 0) = 1.0f;
		for (i = 1; i < N; i++) {
			at(&A, i, 0) = 0.0f;
		}
		printf("P'*L = \n"); matrixf_print(&A, "%9.4f");
		printf("T = \n"); matrixf_print(&T, "%9.4f");
		matrixf_multiply_inplace(&T, &A, &A, 0, 1, work);
		printf("P'*L*T*L'*P = \n"); matrixf_print(&T, "%9.4f");
	}

	for (i = 0; i < N * N; i++) {
		copy[i] -= T.data[i];
	}
	printf("\n||e|| = %g\n", normf(copy, N * N, 1));

	return 0;
}