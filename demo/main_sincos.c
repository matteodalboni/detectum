#include <stdio.h>
#include "detectum.h"

#define n 4

int main()
{
	float A_data[] = {
		16,  2,  3, 13,
		 5, 11, 10,  8,
		 9,  7,  6, 12,
		 4, 14, 15,  1
	};
	float work[3 * (2 * n) * (2 * n) + 2 * n];
	int i, j;
	Matrixf A;
	Matrixf(Z, 2 * n, 2 * n);
	Matrixf(S, n, n);
	Matrixf(C, n, n);
	Matrixf(I, n, n);

	matrixf_init(&A, n, n, A_data, 1);
	printf("A =\n"); matrixf_print(&A, "%9.4f");
	for (j = 0; j < n; j++) {
		for (i = 0; i < n; i++) {
			at(&Z, i + n, j) = +at(&A, i, j);
			at(&Z, i, j + n) = -at(&A, i, j);
		}
	}
	matrixf_exp(&Z, work);
	for (j = 0; j < n; j++) {
		for (i = 0; i < n; i++) {
			at(&S, i, j) = at(&Z, i + n, j);
			at(&C, i, j) = at(&Z, i + 0, j);
		}
	}
	printf("\nsin(A) =\n"); matrixf_print(&S, "%9.4f");
	printf("\ncos(A) =\n"); matrixf_print(&C, "%9.4f");
	matrixf_multiply(&S, &S, &I, 1, 0, 0, 0);
	matrixf_multiply(&C, &C, &I, 1, 1, 0, 0);
	printf("\nsin(A)^2 + cos(A)^2 =\n"); matrixf_print(&I, "%9.4f");
	return 0;
}