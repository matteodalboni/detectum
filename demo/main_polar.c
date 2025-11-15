#include <stdio.h>
#include "detectum.h"

#define m 5
#define n 4

int main()
{
	float A_data[] = {
		17, 24,  1,  8,
		23,  5,  7, 14,
		 4,  6, 13, 20,
		10, 12, 19, 21,
		11, 18, 25,  2
	};
	float work[n];
	Matrixf(U, m, n);
	Matrixf(V, n, n);
	Matrixf(R, m, n);
	Matrixf A;

	matrixf_init(&A, m, n, A_data, 1);
	printf("A =\n"); matrixf_print(&A, "%9.4f");
	matrixf_decomp_svd(&A, &U, &V);
	matrixf_multiply_inplace(&U, 0, &V, 0, 1, work); // U <-- U*V'
	matrixf_multiply_inplace(&A, &V, &V, 0, 1, work); // A <-- V*A*V'
	printf("\nThe polar decomposition of A is U*P, where\n");
	printf("\nU =\n"); matrixf_print(&U, "%9.4f");
	printf("\nP =\n"); matrixf_print(&A, "%9.4f");

	return 0;
}