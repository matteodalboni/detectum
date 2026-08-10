#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

int matrixf_get_eigenvector_it(Matrixf* A, Matrixf* v,
	float lamre, float lamim, int iter, float* work)
{
	int i, j;
	const int n = A->rows;
	const int p = lamim == 0 ? 1 : 2;
	const int q = v->cols;
	float norm, eps = epsf(lamre);
	Matrixf perm = { p * n, 1, work };
	Matrixf C = { p * n, p * n, work + p * n };

	if (A->cols != n || q > 2 ||
		v->rows != n || q < p) {
		return -1;
	}
	for (j = 0; j < n; j++) {
		for (i = 0; i < n; i++) {
			at(&C, i, j) = at(A, i, j);
			if (lamim != 0) {
				at(&C, i + n, j + n) = at(A, i, j);
				at(&C, i + n, j) = 0;
				at(&C, i, j + n) = 0;
			}
		}
		at(&C, j, j) -= lamre + eps;
		if (lamim != 0) {
			at(&C, j + n, j + n) -= lamre + eps;
			at(&C, j + n, j) = -lamim;
			at(&C, j, j + n) = +lamim;
		}
		else if (q == 2) {
			v->data[j + n] = 0;
		}
	}
	v->rows = p * n;
	v->cols = 1;
	matrixf_decomp_lu(&C, &perm, 0);
	for (i = 0; i < iter; i++) {
		matrixf_permute(v, &perm, 0, 0);
		matrixf_solve_tril(&C, v, v, 1);
		matrixf_solve_triu(&C, v, v, 0);
		norm = normf(v->data, p * n, 1);
		for (j = 0; j < p * n; j++) {
			v->data[j] /= norm;
		}
	}
	v->rows = n;
	v->cols = q;
	return 0;
}

#define n 4

int main()
{
	int i, k = 0;
	float eigval_re = 0, eigval_im = 0;
	float work[4 * n * n + 2 * n] = { 0 };
	float A_data[] = {
#if 1
		 1, 1, 1, 3,
		 1, 2, 1, 1,
		 1, 1, 3, 1,
		-2, 1, 1, 4
#else // defective matrix
		3, 1, 0, 0,
		0, 3, 1, 0,
		0, 0, 3, 1,
		0, 0, 0, 3
#endif
	};
	Matrixf A;
	Matrixf(T, n, n);
	Matrixf(v, n, 2);

	matrixf_init(&A, n, n, A_data, 1);
	for (i = 0; i < n * n; i++) T.data[i] = A_data[i];
	matrixf_decomp_schur(&T, 0);
	printf("\nA = [\n"); matrixf_print(&A, "%9.4f "); printf("];\n");
	printf("\nT = [\n"); matrixf_print(&T, "%9.4f "); printf("];\n");
	for (k = 0; k < n; k++) {
		eigval_re = at(&T, k, k);
		eigval_im = 0;
		if (k > 0 && at(&T, k, k - 1) != 0)
			eigval_im = -sqrtf(-at(&T, k - 1, k) * at(&T, k, k - 1));
		if (k < n - 1 && at(&T, k + 1, k) != 0)
			eigval_im = +sqrtf(-at(&T, k + 1, k) * at(&T, k, k + 1));
		for (i = 0; i < 2 * n; i++) {
			v.data[i] = (float)rand() / (float)RAND_MAX;
		}
		matrixf_get_eigenvector_it(&A, &v, eigval_re, eigval_im, 2, work);
		printf("\neigval(:,%d) = %0.4f%+.4fi;\neigvec(:,%d) = [\n",
			k + 1, eigval_re, eigval_im, k + 1);
		for (i = 0; i < n; i++)
			printf("   %9.4f%+.4fi\n", v.data[i], v.data[i + n]);
		printf("];\n");
	}
	return 0;
}