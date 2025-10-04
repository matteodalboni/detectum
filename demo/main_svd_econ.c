#include <stdio.h>
#include "detectum.h"

int main()
{
#if 0
#define m 5
#define n 4
	float A_data[] = {
		17, 24,  1,  8,
		23,  5,  7, 14,
		 4,  6, 13, 20,
		10, 12, 19, 21,
		11, 18, 25,  2
	};
	Matrixf(U, m, n);
	Matrixf(V, n, n);
#else
#define m 4
#define n 5
	float A_data[] = {
		17, 24,  1,  8, 15,
		23,  5,  7, 14, 16,
		 4,  6, 13, 20, 22,
		10, 12, 19, 21,  3
	};
	Matrixf(U, m, m);
	Matrixf(V, n, m);
#endif
	Matrixf A;

	matrixf_init(&A, m, n, A_data, 1);
	printf("\nA =\n"); matrixf_print(&A, "%9.4f");
	int exitflag = matrixf_decomp_svd(&A, &U, &V);
	printf("\nS =\n"); matrixf_print(&A, "%9.4f");
	printf("\nU =\n"); matrixf_print(&U, "%9.4f");
	printf("\nV =\n"); matrixf_print(&V, "%9.4f");

	return exitflag;
}