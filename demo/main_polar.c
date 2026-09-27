#define DETECTUM_ENABLE_PRINT
#include "detectum.h"

#define m 5
#define n 4

int main()
{
	float A_data[m * n] = {
		17, 24,  1,  8,
		23,  5,  7, 14,
		 4,  6, 13, 20,
		10, 12, 19, 21,
		11, 18, 25,  2
	};
	float work[n];
	size_t i;
	Matrixf(U, m, n);
	Matrixf(V, n, n);
	Matrixf(A_copy, m, n);
	Matrixf A;

	matrixf_init(&A, m, n, A_data, 1);
	for (i = 0; i < m * n; i++) A_copy.data[i] = A_data[i];
	printf("A =\n"); matrixf_print(&A, "%9.4f");
	matrixf_decomp_svd(&A, &U, &V);
	matrixf_multiply_inplace(&U, 0, &V, 0, 1, work); // U <-- U*V'
	matrixf_multiply_inplace(&A, &V, &V, 0, 1, work); // A <-- V*A*V'
	printf("\nThe polar decomposition of A is U*P, where\n");
	printf("\nU =\n"); matrixf_print(&U, "%9.4f");
	printf("\nP =\n"); matrixf_print(&A, "%9.4f");
	matrixf_multiply(&U, &A, &A_copy, 1.0f, -1.0f, 0, 0);
	printf("\n||U*P - A||_F = %g\n", normf(A_copy.data, m * n, 1));

	return 0;
}