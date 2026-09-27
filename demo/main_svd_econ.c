#define DETECTUM_ENABLE_PRINT
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
	Matrixf(US, m, n);
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
	Matrixf(US, m, m);
#endif
	Matrixf(USVt, m, n);
	Matrixf A;
	size_t i;
	float A_data_copy[m * n] = { 0 };

	matrixf_init(&A, m, n, A_data, 1);
	for (i = 0; i < m * n; i++) A_data_copy[i] = A_data[i];
	printf("\nA =\n"); matrixf_print(&A, "%9.4f");
	int exitflag = matrixf_decomp_svd(&A, &U, &V);
	printf("\nU =\n"); matrixf_print(&U, "%9.4f");
	printf("\nS =\n"); matrixf_print(&A, "%9.4f");
	printf("\nV =\n"); matrixf_print(&V, "%9.4f");
	matrixf_multiply(&U, &A, &US, 1.0f, 0.0f, 0, 0);
	matrixf_multiply(&US, &V, &USVt, 1.0f, 0.0f, 0, 1);
	printf("\nU*S*V' =\n"); matrixf_print(&USVt, "%9.4f");
	for (i = 0; i < m * n; i++) A_data_copy[i] -= USVt.data[i];
	printf("\n||U*S*V' - A||_F = %g\n", normf(A_data_copy, m * n, 1));
	return exitflag;
}