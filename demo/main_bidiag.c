#include <stdio.h>
#include <stdlib.h>
#include "detego.h"

static void unpack_bidiag(Matrixf* A, Matrixf* U, Matrixf* V)
{
	int i, j;

	if (A->size[0] < A->size[1]) {
		matrixf_transpose(A);
		unpack_bidiag(A, V, U);
		matrixf_transpose(A);
		return;
	}
	for (i = 0; i < U->size[1]; i++) {
		at(U, i, i) = 1;
	}
	for (i = 0; i < V->size[1]; i++) {
		at(V, i, i) = 1;
	}
	matrixf_unpack_householder_bwd(A, U, 0);
	matrixf_transpose(A);
	matrixf_unpack_householder_bwd(A, V, 1);
	matrixf_transpose(A);
	if (U->size[0] == U->size[1]) {
		matrixf_transpose(A);
		A->size[1] = U->size[0];
		matrixf_transpose(A);
	}
	for (j = 0; j < A->size[1]; j++) {
		for (i = 0; i < A->size[0]; i++) {
			if (i != j && (i + 1) != j) {
				at(A, i, j) = 0;
			}
		}
	}
}

#define ONE_STEP 0
#define m 10
#define n 5

int main()
{
	float A_data[m * n] = {
		92, 99,   1,  8, 15,
		98, 80,   7, 14, 16,
		 4, 81,  88, 20, 22,
		85, 87,  19, 21,  3,
		86, 93,  25,  2,  9,
		17, 24,  76, 83, 90,
		23,  5,  82, 89, 91,
		79,  6,  13, 95, 97,
		10, 12,  94, 96, 78,
		11, 18, 100, 77, 84
	};
	Matrixf A, C;
	Matrixf U = matrixf(m, m);
	Matrixf V = matrixf(n, n);

	matrixf_init(&A, m, n, A_data, 1);
	printf("A = [\n"); matrixf_print(&A, "%9.4f "); printf("];\n\n");
#if ONE_STEP
	matrixf_decomp_bidiag(&A, &U, &V);
#else
	matrixf_decomp_bidiag(&A, 0, 0);
	unpack_bidiag(&A, &U, &V);
#endif
	printf("B = [\n"); matrixf_print(&A, "%9.4f "); printf("];\n\n");
	printf("U = [\n"); matrixf_print(&U, "%9.4f "); printf("];\n\n");
	printf("V = [\n"); matrixf_print(&V, "%9.4f "); printf("];\n\n");

	C = matrixf(U.size[0], A.size[1]);
	matrixf_multiply(&U, &A, &C, 1, 0, 0, 0);
	A.size[0] = m;
	A.size[1] = n;
	matrixf_multiply(&C, &V, &A, 1, 0, 0, 1);
	printf("U*B*V' = [\n"); matrixf_print(&A, "%9.4f "); printf("];\n\n");

	free(U.data);
	free(V.data);
	free(C.data);

	return 0;
}