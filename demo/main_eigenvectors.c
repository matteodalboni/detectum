#include "verto.h"
#include "verto_utils.h"
#include "verto_alloc.h"

static void print_complex_eigenvectors(Matrixf* T, Matrixf* V)
{
	int i, j;
	const int n = T->size[0];

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (j == n - 1 || _(T, j + 1, j) == 0) {
				printf("%9.4f ", _(V, i, j));
			}
			else {
				printf("%9.4f%+.4fi ", _(V, i, j), +_(V, i, j + 1));
				printf("%9.4f%+.4fi ", _(V, i, j), -_(V, i, j + 1));
				j++;
			}
		}
		printf("\n");
	}
	printf("\n");
}

#define n 4
#define TOL 1e-5f

int main()
{
	int i;
	Matrixf A = MATRIXF(n, n);
	Matrixf T = MATRIXF(n, n);
	Matrixf U = MATRIXF(n, n);
	Matrixf V = MATRIXF(n, n);
	Matrixf W = MATRIXF(n, n);
	Matrixf D = MATRIXF(n, n);
	if (!A.data || !T.data || !U.data || !V.data || !W.data || !D.data) return -1;
	float work[4 * (n - 2) * (n - 2) + 5] = { 0 };
#if 1
	float A_data[] = {
		 1, 1, 1, 3,
		 1, 2, 1, 1,
		 1, 1, 3, 1,
		-2, 1, 1, 4
	};
	for (i = 0; i < n * n; i++) T.data[i] = A_data[i];
	matrixf_transpose(&T);
#else
	float A_data[n * n] = { 0 };
	FILE* A_file = fopen("../A.bin", "rb");
	fread(A_data, sizeof(float), (size_t)(n * n), A_file); fclose(A_file);
	for (i = 0; i < n * n; i++) T.data[i] = A_data[i];
#endif
	matrixf_decomp_schur(&T, &U);
	matrixf_multiply(&U, &T, &V, 1, 0, 0, 0);
	matrixf_multiply(&V, &U, &A, 1, 0, 0, 1);
	DISP("%9.4f ", A);
	DISP("%9.4f ", T);
	DISP("%9.4f ", U);

	printf("Eigenvectors (compact form):\n\n");
	if (matrixf_get_eigenvectors(&T, &U, &V, &W, 0, work)) return 1;
	DISP("%9.4f ", V);
	DISP("%9.4f ", W);

	printf("Eigenvectors (full form):\n\n");
	printf("V = \n"); print_complex_eigenvectors(&T, &V);
	printf("W = \n"); print_complex_eigenvectors(&T, &W);

	printf("Pseudo-eigenvectors:\n\n");
	if (matrixf_get_eigenvectors(&T, &U, &V, &W, 1, work)) return 1;
	DISP("%9.4f ", V);
	DISP("%9.4f ", W);
	
	printf("If V is invertible, block-diagonalize A:\n\n");
	matrixf_multiply(&A, &V, &D, 1, 0, 0, 0);
	matrixf_solve_lu(&V, &D);
	for (i = 0; i < n * n; i++)
		if (fabsf(D.data[i]) < TOL) D.data[i] = 0;
	printf("inv(V)*A*V = "); DISP("%9.4g ", D);
	
	printf("If W is invertible, block-diagonalize A:\n\n");
	matrixf_multiply(&A, &W, &D, 1, 0, 1, 0);
	matrixf_solve_lu(&W, &D); matrixf_transpose(&D);
	for (i = 0; i < n * n; i++)
		if (fabsf(D.data[i]) < TOL) D.data[i] = 0;
	printf("W'*A*inv(W') = "); DISP("%9.4g ", D);

	free(A.data);
	free(T.data);
	free(U.data);
	free(V.data);
	free(W.data);
	free(D.data);

	return 0;
}