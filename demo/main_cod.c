#define DETECTUM_ENABLE_PRINT
#define DETECTUM_ENABLE_ALLOC
#include "detectum.h"

#define ONE_STEP
#define m 10
#define n 5

int main()
{
	float A_data[m * n] = {
		92, 0, 0, 0, 15,
		98, 0, 0, 0, 16,
		 4, 0, 0, 0, 22,
		85, 0, 0, 0,  3,
		86, 0, 0, 0,  9,
		17, 0, 0, 0, 90,
		23, 0, 0, 0, 91,
		79, 0, 0, 0, 97,
		10, 0, 0, 0, 78,
		11, 0, 0, 0, 84
	};
	float A_data_copy[m * n] = { 0 };
	int rank;
	size_t i = 0, j = 0;
	Matrixf A;
	Matrixf P = matrixf(1, n);
	Matrixf U = matrixf(m, m);
	Matrixf V = matrixf(n, n);
	Matrixf UT = matrixf(m, n);

	// Initialization
	matrixf_init(&A, m, n, A_data, 1);
	for (i = 0; i < m * n; i++) A_data_copy[i] = A_data[i];
	printf("A = [\n"); matrixf_print(&A, "%10.5g "); printf("];\n\n");
	// Complete orthogonal decomposition (COD)
#ifdef ONE_STEP
	rank = matrixf_decomp_cod(&A, &U, &V, &P, -1);
#else
	rank = matrixf_decomp_cod(&A, &U, 0, &P, -1);
	matrixf_transpose(&A);
	A.cols = rank;
	for (i = 0; i < n; i++) {
		at(&V, i, i) = 1;
	}
	matrixf_unpack_house(&A, &V, 0, 0);
	for (j = 0; j < rank; j++) {
		for (i = j + 1; i < A.rows; i++) {
			at(&A, i, j) = 0;
		}
	}
	A.cols = m;
	matrixf_transpose(&A);
#endif
	matrixf_permute(&V, &P, 1, 0);
	printf("The rank of A is %d\n\n", rank);
	printf("U = [\n"); matrixf_print(&U, "%10.5g "); printf("];\n\n");
	printf("T = [\n"); matrixf_print(&A, "%10.5g "); printf("];\n\n");
	printf("P*V = [\n"); matrixf_print(&V, "%10.5g "); printf("];\n\n");
	// Matrix reconstruction
	matrixf_multiply(&U, &A, &UT, 1, 0, 0, 0);
	A.rows = m; A.cols = n;
	matrixf_multiply(&UT, &V, &A, 1, 0, 0, 1);
	printf("U*T*(P*V)' = [\n"); matrixf_print(&A, "%10.5g "); printf("];\n\n");
	for (i = 0; i < m * n; i++) A_data_copy[i] -= A_data[i];
	printf("||U*T*(P*V)' - A||_F = %g\n", normf(A_data_copy, m * n, 1));
	// Memory release
	free(P.data);
	free(U.data);
	free(V.data);
	free(UT.data);

	return 0;
}