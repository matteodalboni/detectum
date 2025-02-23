#include "verto.h"
#include "verto_utils.h"
#include "verto_alloc.h"

int main()
{
	int i, j;
	Matrixf A = MATRIXF(5, 7);
	Matrixf B = MATRIXF(A.size[0], 2);
	Matrixf X = MATRIXF(A.size[1], B.size[1]);
	FILE* A_file = fopen("../A.bin", "rb");

	if (!A.data) return -1;
	fread(A.data, sizeof(float), (size_t)(A.size[0] * A.size[1]), A_file);
	fclose(A_file);
	for (j = 0; j < B.size[1]; j++)
		for (i = 0; i < B.size[0]; i++) 
			_(&B, i, j) = j + 1.0f;
	DISP("%9.4f ", A);
	DISP("%9.4f ", B);

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
	DISP("%9.4f ", X);

	free(A.data);
	free(B.data);
	free(X.data);

	return 0;
}