#include <stdio.h>
#include "detectum.h"

int matrixf_decomp_ldl(Matrixf* A)
{
	const int n = A->rows;
	int i, j, k;
	float a, * Ai, * Aj;

	for (j = 0; j < n; j++) {
		Aj = &at(A, 0, j);
		for (i = 0; i < j; i++) {
			Ai = &at(A, 0, i);
			a = Ai[j] * at(A, i, i);
			for (k = j; k < n; k++) {
				Aj[k] -= Ai[k] * a;
			}
		}
		a = at(A, j, j);
		if (a != 0) {
			for (k = j; k < n; k++) {
				if (k > j) {
					Aj[k] /= a;
				}
			}
		}
	}
	return 0;
}

#define n 6

int main()
{
	float A_data[n * n] = {
		 92,   85,   -4, -60,  -40,  55,
		 85,  147,   44, -84, -111,  97,
		 -4,   44,  152,  86, -125,  23,
		-60,  -84,   86, 334,   -4, -99,
		-40, -111, -125,  -4,  173, -93,
		 55,   97,   23, -99,  -93,  93
	};
	float L_data[n * n] = { 0 };
	float D_data[n * n] = { 0 };
	Matrixf A = { n, n, A_data };
	Matrixf L = { n, n, L_data };
	Matrixf D = { n, n, D_data };
	int i, j;

	matrixf_init(&A, n, n, A_data, 1);
	printf("\nA = \n"); matrixf_print(&A, "%9.4f ");
	printf("\nLDL decomposition\n");
	matrixf_decomp_ldl(&A);
	for (j = 0; j < n; j++) {
		at(&D, j, j) = at(&A, j, j);
		at(&L, j, j) = 1;
		for (i = j + 1; i < n; i++)
			at(&L, i, j) = at(&A, i, j);
	}
	printf("\nL = \n"); matrixf_print(&L, "%9.4f ");
	printf("\nD = \n"); matrixf_print(&D, "%9.4f ");
	matrixf_multiply_inplace(&D, &L, &L, 0, 1, A_data);
	printf("\nL*D*L' = \n"); matrixf_print(&D, "%9.4f ");

	printf("\nCholesky decomposition\n");
	if (matrixf_decomp_chol(&D)) {
		printf("The matrix is not positive definite!\n");
		return 1;
	}
	for (j = 0; j < n - 1; j++)
		for (i = j + 1; i < n; i++)
			at(&D, i, j) = 0;
	printf("\nR = \n"); matrixf_print(&D, "%9.4f ");
	matrixf_multiply(&D, &D, &L, 1, 0, 1, 0);
	printf("\nR'*R = \n"); matrixf_print(&L, "%9.4f ");
	return 0;
}