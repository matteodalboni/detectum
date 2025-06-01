#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

static int inverse_iteration(Matrixf* A, Matrixf* v,
	float eigval_re, float eigval_im, float* work, int iter)
{
	int i, j;
	const int n = A->size[0];
	float norm;
	Matrixf p = { { 2 * n, 1 }, work };
	Matrixf C = { { 2 * n, 2 * n }, work + 2 * n };

	for (j = 0; j < n; j++) {
		for (i = 0; i < n; i++) {
			at(&C, i, j) = at(&C, i + n, j + n) = at(A, i, j);
			at(&C, i + n, j) = at(&C, i, j + n) = 0;
		}
		at(&C, j, j) -= eigval_re;
		at(&C, j + n, j + n) -= eigval_re;
		at(&C, j + n, j) = -eigval_im;
		at(&C, j, j + n) = +eigval_im;
		p.data[j] = (float)j;
		p.data[j + n] = (float)(j + n);
		v->data[j] = 1;
		v->data[j + n] = 0;
	}
	v->size[0] = 2 * n; 
	v->size[1] = 1;
	matrixf_decomp_lu(&C, &p);
	for (i = 0; i < iter; i++) {
		matrixf_permute(v, &p, 0);
		matrixf_solve_tril(&C, v, v, 1);
		matrixf_solve_triu(&C, v, v, 0);
		norm = normf(v->data, 2 * n, 1);
		for (j = 0; j < 2 * n; j++) {
			v->data[j] /= norm;
		}
	}
	v->size[0] = n;
	v->size[1] = 2;
	return 0;
}

#define n 4

int main()
{
	int i, k = 0;
	float eigval_re = 0, eigval_im = 0;
	float work[4 * n * n + 2 * n] = { 0 };
	float A_data[] = {
		 1, 1, 1, 3,
		 1, 2, 1, 1,
		 1, 1, 3, 1,
		-2, 1, 1, 4
	};
	Matrixf A;
	Matrixf T = matrixf(n, n);
	Matrixf v = matrixf(n, 2);

	if (!T.data || !v.data) return -1;

	matrixf_init(&A, n, n, A_data, 1);
	for (i = 0; i < n * n; i++) T.data[i] = A_data[i];
	matrixf_decomp_schur(&T, 0);
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	printf("T = \n"); matrixf_print(&T, "%9.4f "); printf("\n");

	for (k = 0; k < n; k++) {
		eigval_re = at(&T, k, k);
		eigval_im = 0;
		if (k > 0 && at(&T, k, k - 1) != 0)
			eigval_im = -sqrtf(-at(&T, k - 1, k) * at(&T, k, k - 1));
		if (k < n - 1 && at(&T, k + 1, k) != 0)
			eigval_im = +sqrtf(-at(&T, k + 1, k) * at(&T, k, k + 1));
		inverse_iteration(&A, &v, eigval_re, eigval_im, work, 2);
		printf("\n%d) eigval = %0.4f%+.4fi\n   eigvec = \n",
			k + 1, eigval_re, eigval_im);
		for (i = 0; i < n; i++)
			printf("   %9.4f%+.4fi\n", v.data[i], v.data[i + n]);
	}

	free(T.data);
	free(v.data);

	return 0;
}