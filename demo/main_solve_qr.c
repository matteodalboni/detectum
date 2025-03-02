#define DETEGO_USE_PRINT
#define DETEGO_USE_ALLOC
#include "detego.h"

int main()
{
	int i, j;
	Matrixf A = matrixf(5, 7);
	Matrixf B = matrixf(A.size[0], 2);
	Matrixf X = matrixf(A.size[1], B.size[1]);
	FILE* A_file = fopen("../A.bin", "rb");

	if (!A.data) return -1;
	fread(A.data, sizeof(float), (size_t)(A.size[0] * A.size[1]), A_file);
	fclose(A_file);
	for (j = 0; j < B.size[1]; j++)
		for (i = 0; i < B.size[0]; i++) 
			at(&B, i, j) = j + 1.0f;
	printf("A = \n"); matrixf_print(&A, "%9.4f ");
	printf("B = \n"); matrixf_print(&B, "%9.4f ");

#if 0
	// Minimum norm solution to full-rank underdetermined system
	if (A.size[0] < A.size[1]) {
		matrixf_transpose(&A);
		matrixf_decomp_qr(&A, 0, 0, 0);
		matrixf_transpose(&A);
		matrixf_solve_tril(&A, &B, &X, 0);
		matrixf_transpose(&A);
		matrixf_accumulate_bwd(&A, &X);
	}
#else
	matrixf_solve_qr(&A, &B, &X, -1);
#endif
	printf("X = \n"); matrixf_print(&X, "%9.4f ");

	free(A.data);
	free(B.data);
	free(X.data);

	return 0;
}