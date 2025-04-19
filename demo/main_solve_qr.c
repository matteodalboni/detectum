#include <stdio.h>
#include <stdlib.h>
#include "detego.h"

#define IN_PLACE

int main()
{
	int i, j;
	float work[7] = { 0 };
	FILE* A_file = fopen("../A.bin", "rb");
	Matrixf A = matrixf(5, 7);
#ifdef IN_PLACE
	float* data = malloc(sizeof(float) * 7 * 2);
	Matrixf B = { { A.size[0], 2 }, data };
	Matrixf X = { { A.size[1], 2 }, data };
#else
	Matrixf B = matrixf(A.size[0], 2);
	Matrixf X = matrixf(A.size[1], 2);
#endif

	if (!A.data) return -1;
	fread(A.data, sizeof(float), (size_t)(A.size[0] * A.size[1]), A_file);
	fclose(A_file);
	for (j = 0; j < B.size[1]; j++)
		for (i = 0; i < B.size[0]; i++)
			at(&B, i, j) = j + 1.0f;
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	printf("B = \n"); matrixf_print(&B, "%9.4f "); printf("\n");
	matrixf_solve_qrp(&A, &B, &X, -1, work);
	printf("X = \n"); matrixf_print(&X, "%9.4f "); printf("\n");

	free(A.data);
	free(B.data);
#ifndef IN_PLACE
	free(X.data);
#endif

	return 0;
}