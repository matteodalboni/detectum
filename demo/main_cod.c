#include <stdio.h>
#include <stdlib.h>
#include "detego.h"

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
	Matrixf A;
	Matrixf p = matrixf(1, n);
	Matrixf U = matrixf(m, m);
	Matrixf V = matrixf(n, n);
	Matrixf UT = matrixf(m, n);
	const int q = m < n ? m : n;
	int rank = 0;
	float tol = 0;

	// Initialization
	matrixf_init(&A, m, n, A_data, 1);
	printf("A = [\n"); matrixf_print(&A, "%10.5g "); printf("];\n\n");
	// Complete orthogonal decomposition (COD)
	matrixf_decomp_qr(&A, &U, &p, 0);
	tol = (m > n ? m : n) * epsf(at(&A, 0, 0));
	while (rank < q && fabsf(at(&A, rank, rank)) > tol) {
		rank++;
	}
	printf("The rank of A is %d\n\n", rank);
	matrixf_transpose(&A);
	A.size[1] = rank;
	matrixf_decomp_qr(&A, &V, 0, 0);
	A.size[1] = m;
	matrixf_transpose(&A);
	matrixf_permute(&V, &p, 1);
	printf("U = [\n"); matrixf_print(&U, "%10.5g "); printf("];\n\n");
	printf("T = [\n"); matrixf_print(&A, "%10.5g "); printf("];\n\n");
	printf("V = [\n"); matrixf_print(&V, "%10.5g "); printf("];\n\n");
	// Matrix reconstruction
	matrixf_multiply(&U, &A, &UT, 1, 0, 0, 0);
	matrixf_multiply(&UT, &V, &A, 1, 0, 0, 1);
	printf("U*T*V' = [\n"); matrixf_print(&A, "%10.5g "); printf("];\n\n");
	// Memory release
	free(p.data);
	free(U.data);
	free(V.data);
	free(UT.data);

	return 0;
}