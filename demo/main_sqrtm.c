#include <stdio.h>
#include "detectum.h"

#define n 10

int main() 
{
	int exitflag;
	float work[n * n + n] = { 0 };
	Matrixf A, XX = { { n, n }, work };
#if 0
	float A_data[n * n] = { 0 };
	FILE* A_file = fopen("../A.bin", "rb");
	fread(A_data, sizeof(float), n * n, A_file); fclose(A_file);
	matrixf_init(&A, n, n, A_data, 0);
#else
	float A_data[n * n] = {
		7,  2,  1,  5,  7,  8, -3, -4,   3,  1,
	    0, -2, -4,  0,  6, -9,  7,  3,  -1, -2,
	   -1, -4,  4, 11,  8, -1, 12, -2,   4,  1,
        1,  4, -4, -6,  1,  4,  5, -3,  -9, -5,
        3, -4, -7,  6,  0,  4,  4,  4,  -8, 10,
       -7, -2,-10, -9, -5,  4,  4, 10,  -8,  7,
        7,  1,  6,  6, -4, -1,  1, 10,   6,  9,
       -8, -4, -5,  2, -8, -9, -6,  1,  -7, -7,
       -8, 12,  5,  4, -7,  0,  9, -1,   1, -7,
      -10,  7,  5, -9,  3, -7, -2,  8, -11, -2
	};
	matrixf_init(&A, n, n, A_data, 1);
#endif
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	exitflag = matrixf_sqrt(&A, work);
	printf("X = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	matrixf_multiply(&A, &A, &XX, 1, 0, 0, 0);
	printf("X*X = \n"); matrixf_print(&XX, "%9.4f "); printf("\n");

	return exitflag;
}