#include <stdio.h>
#include "detectum.h"

#define M 5
#define N 4
#define P 2

int main()
{
	float A_data[M * N] = {
		17, 24, 1, 8,
		23, 5, 7, 14,
		4, 6, 13, 20,
		10, 12, 19, 21,
		11, 18, 25, 2
	};
	float B_data[M * P] = {
		15, 3 * 15,
		16, 3 * 16,
		22, 3 * 22,
		 3, 3 * 3,
		 9, 3 * 9
	};
	Matrixf A = { 0 }, B = { 0 }, X = { 0 };

	matrixf_init(&A, M, N, A_data, 1);
	matrixf_init(&B, M, P, B_data, 1);
	matrixf_init(&X, N, P, B_data, 0);
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	printf("B = \n"); matrixf_print(&B, "%9.4f "); printf("\n");
	matrixf_solve_qr(&A, &B, &X);
	printf("The (least-squares) solution of the linear system A*X = B is:\n\n");
	printf("X = \n"); matrixf_print(&X, "%9.4f "); printf("\n");

	return 0;
}