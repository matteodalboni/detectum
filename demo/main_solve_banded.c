#include <stdio.h>
#include "detectum.h"

#define n 7
#define p 2
#define ubw 3

int main()
{
	int i, j;
	float X_data[n * p] = { 0 };
	float C_data[n * n] = { 0 };
	Matrixf A, B;
	Matrixf X = { n, p, X_data };
	Matrixf C = { n, n, C_data };
#if 0
	float A_data[n * n] = { 0 };
	float B_data[n * p] = { 0 };
	FILE* A_file = fopen("../Aband.bin", "rb");
	FILE* B_file = fopen("../B.bin", "rb");
	fread(A_data, sizeof(float), n * n, A_file); fclose(A_file);
	fread(B_data, sizeof(float), n * p, B_file); fclose(B_file);
	matrixf_init(&A, n, n, A_data, 0);
	matrixf_init(&B, n, p, B_data, 0);
#else
	float A_data[n * n] = {	
		 -7, -2,  1,  0,  -2, -6,  7,
		 -2, -5,  6,  3,  10, -2, -2,
		  0, -3,  2,  0,   0, -5, -2,
		  0,  0,  9,  3,   2, -9, 10,
		  0,  0,  0, -6, -10, -6, -3,
		  0,  0,  0,  0,   6,  9, -8,
		  0,  0,  0,  0,   0,  1,  4
	};
	float B_data[n * p] = {
		  2, -1,
		-10, -2,
		  9, 10,
		  2,  0,
		  4, -8,
		 -2,  8,
		  4, -3
	};
	matrixf_init(&A, n, n, A_data, 1);
	matrixf_init(&B, n, p, B_data, 1);
	for (j = 1; j < n; j++) {
		for (i = 0; i < j - ubw; i++) {
			at(&A, i, j) = 0;
		}
	}
#endif
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	printf("B = \n"); matrixf_print(&B, "%9.4f "); printf("\n");
	for (i = 0; i < n * p; i++) X.data[i] = B.data[i];
	for (i = 0; i < n * n; i++) C.data[i] = A.data[i];
	if (matrixf_solve_lu_banded(&A, &X, ubw)) return -1;
	printf("X = \n"); matrixf_print(&X, "%9.4f "); printf("\n");
	matrixf_multiply(&C, &X, &B, 1, -1, 0, 0);
	printf("||A*X - B||_F = %g\n", normf(B.data, n * p, 1));

	return 0;
}