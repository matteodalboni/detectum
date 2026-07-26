#include <stdio.h>
#include "detectum.h"

#define n 6
#define FULL_P

int main()
{
	int i;
#if 0
	float A_data[n * n] = { 0 };
	FILE* A_file = fopen("../A.bin", "rb");
	fread(A_data, sizeof(float), n * n, A_file);
	fclose(A_file);
#else
	float A_data[n * n] = {
		 92,   85,   -4, -60,  -40,  55,
		 85, -147,   44, -84, -111,  97,
		 -4,   44,  152,  86, -125,  23,
		-60,  -84,   86, 334,   -4, -99,
		-40, -111, -125,  -4,  173, -93,
		 55,   97,   23, -99,  -93,  93
	};
#endif
	float A_copy[n * n] = { 0 };
	float P_data[n * n] = { 0 };
	float perm_data[n * n] = { 0 };
	float T_data[n * n] = { 0 };
	float B_data[n * n] = { 0 };
	Matrixf A = { n, n, A_data };
	Matrixf P = { n, n, P_data };
	Matrixf perm = { n, 1, perm_data };
	Matrixf T = { n, n, T_data };
	Matrixf B = { n, n, B_data };

	printf("Input matrix:\n");
	printf("\nA = \n"); matrixf_print(&A, "%9.4f ");
	for (i = 0; i < n * n; i++) A_copy[i] = A_data[i];
	matrixf_decomp_ltl(&A);
	printf("\nPacked form:\n");
	printf("\nA = \n"); matrixf_print(&A, "%9.4f ");
	printf("\nUnpacked form:\n");
	P.data[0] = 1;
	for (i = 0; i < n; i++) {
		if (i > 0) {
			at(&P, i, (int)at(&A, i, 0)) = 1;
			at(&perm, i, 0) = at(&A, i, 0);
			at(&A, i, 0) = 0;
		}
		at(&T, i, i) = at(&A, i, i);
		at(&A, i, i) = 1;
		if (i < n - 1) {
			at(&T, i + 1, i) = at(&A, i, i + 1);
			at(&T, i, i + 1) = at(&A, i, i + 1);
			at(&A, i, i + 1) = 0;
		}
	}
	printf("\nL = \n"); matrixf_print(&A, "%9.4f ");
	printf("\nT = \n"); matrixf_print(&T, "%9.4f ");
#ifdef FULL_P
	printf("\nP = \n"); matrixf_print(&P, "%3.0f ");
	matrixf_multiply(&P, &A, &B, 1, 0, 1, 0);
	matrixf_multiply(&B, &T, &P, 1, 0, 0, 0);
	matrixf_multiply(&P, &B, &A, 1, 0, 0, 1);
#else
	printf("\nperm = \n"); matrixf_print(&perm, "%3.0f ");
	for (i = 0; i < n * n; i++) {
		B.data[i] = A.data[i];
		A.data[i] = T.data[i];
	}
	matrixf_permute(&B, &perm, 0, 1);
	printf("\nP'*L = \n"); matrixf_print(&B, "%9.4f ");
	matrixf_multiply_inplace(&A, &B, &B, 0, 1, perm.data);
#endif
	printf("\nReconstructed matrix:\n");
	printf("\nP'*L*T*L'*P = \n"); matrixf_print(&A, "%9.4f ");
	for (i = 0; i < n * n; i++) A_copy[i] -= A.data[i];
	printf("\n||A - P'*L*T*L'*P||_F = %g\n", normf(A_copy, n * n, 1));

	return 0;
}