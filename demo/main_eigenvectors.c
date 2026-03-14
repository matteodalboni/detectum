#include <stdlib.h>
#include <stdio.h>
#include "detectum.h"

static void print_complex_eigenvectors(Matrixf* T, Matrixf* V)
{
	int i, j;
	const int n = T->rows;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (j == n - 1 || at(T, j + 1, j) == 0) {
				printf("%9.4f ", at(V, i, j));
			}
			else {
				printf("%9.4f%+.4fi ", at(V, i, j), +at(V, i, j + 1));
				printf("%9.4f%+.4fi ", at(V, i, j), -at(V, i, j + 1));
				j++;
			}
		}
		printf("\n");
	}
}

#define n 4
#define TOL 1e-5f

int main()
{
	int i;
	Matrixf A = matrixf(n, n);
	Matrixf T = matrixf(n, n);
	Matrixf U = matrixf(n, n);
	Matrixf V = matrixf(n, n);
	Matrixf W = matrixf(n, n);
	Matrixf D = matrixf(n, n);
	if (!A.data || !T.data || !U.data || !V.data || !W.data || !D.data) return -1;
	float work[4 * (n - 2) * (n - 2) + 2] = { 0 };
#if 1
	float A_data[] = {
		 1, 1, 1, 3,
		 1, 2, 1, 1,
		 1, 1, 3, 1,
		-2, 1, 1, 4
	};
	for (i = 0; i < n * n; i++) T.data[i] = A_data[i];
	matrixf_transpose(&T);
#else
	float A_data[n * n] = { 0 };
	FILE* A_file = fopen("../A.bin", "rb");
	fread(A_data, sizeof(float), (size_t)(n * n), A_file); fclose(A_file);
	for (i = 0; i < n * n; i++) T.data[i] = A_data[i];
#endif
	matrixf_decomp_schur(&T, &U);
	matrixf_multiply(&U, &T, &V, 1, 0, 0, 0);
	matrixf_multiply(&V, &U, &A, 1, 0, 0, 1);
	printf("\nA = \n"); matrixf_print(&A, "%9.4f ");
	printf("\nT = \n"); matrixf_print(&T, "%9.4f ");
	printf("\nU = \n"); matrixf_print(&U, "%9.4f ");

	printf("\n\nEigenvectors (compact form):\n");
	if (matrixf_get_eigenvectors(&T, &U, &V, &W, 0, work)) return 1;
	printf("\nV = \n"); matrixf_print(&V, "%9.4f ");
	printf("\nW = \n"); matrixf_print(&W, "%9.4f ");

	printf("\n\nEigenvectors (full form):\n");
	printf("\nV = \n"); print_complex_eigenvectors(&T, &V);
	printf("\nW = \n"); print_complex_eigenvectors(&T, &W);

	printf("\n\nPseudo-eigenvectors:\n");
	if (matrixf_get_eigenvectors(&T, &U, &V, &W, 1, work)) return 1;
	printf("\nV = \n"); matrixf_print(&V, "%9.4f ");
	printf("\nW = \n"); matrixf_print(&W, "%9.4f ");
	
	printf("\nIf V is invertible, block-diagonalize A:\n");
	matrixf_multiply(&A, &V, &D, 1, 0, 0, 0);
	matrixf_solve_lu(&V, &D);
	for (i = 0; i < n * n; i++)
		if (fabsf(D.data[i]) < TOL) D.data[i] = 0;
	printf("\ninv(V)*A*V = \n"); matrixf_print(&D, "%9.4g ");
	
	printf("\nIf W is invertible, block-diagonalize A:\n");
	matrixf_multiply(&A, &W, &D, 1, 0, 1, 0);
	matrixf_solve_lu(&W, &D); matrixf_transpose(&D);
	for (i = 0; i < n * n; i++)
		if (fabsf(D.data[i]) < TOL) D.data[i] = 0;
	printf("\nW'*A*inv(W') = \n"); matrixf_print(&D, "%9.4g ");

	free(A.data);
	free(T.data);
	free(U.data);
	free(V.data);
	free(W.data);
	free(D.data);

	return 0;
}