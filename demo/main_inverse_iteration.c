#define DETEGO_USE_PRINT
#define DETEGO_USE_ALLOC
#include "detego.h"

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
	}
	matrixf_decomp_lu(&C, &p);
	for (i = 0; i < iter; i++) {
		matrixf_permute(v, &p, 0);
		matrixf_solve_tril(&C, v, v, 1);
		matrixf_solve_triu(&C, v, v, 0);
		norm = get_norm2(v->data, 2 * n, 1);
		for (j = 0; j < 2 * n; j++) {
			v->data[j] /= norm;
		}
	}
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
	Matrixf H = matrixf(n, n);
	Matrixf T = matrixf(n, n);
	Matrixf P = matrixf(n, n);
	Matrixf x = matrixf(2 * n, 1);
	Matrixf v = { { n, 2 }, work };

	if (!H.data || !T.data || !P.data || !x.data) return -1;

	for (i = 0; i < n * n; i++) H.data[i] = A_data[i];
	matrixf_transpose(&H);
	matrixf_decomp_hess(&H, &P);
	for (i = 0; i < n * n; i++) T.data[i] = H.data[i];
	disp(H, "%9.4f ");
	matrixf_decomp_schur(&T, 0);
	disp(T, "%9.4f ");

	for (k = 0; k < n; k++) {
		x.size[0] = 2 * n; x.size[1] = 1;
		for (i = 0; i < 2 * n; i++) x.data[i] = (float)(i < n);
		eigval_re = at(&T, k, k);
		eigval_im = 0;
		if (k > 0 && at(&T, k, k - 1) != 0)
			eigval_im = -sqrtf(-at(&T, k - 1, k) * at(&T, k, k - 1));
		if (k < n - 1 && at(&T, k + 1, k) != 0)
			eigval_im = +sqrtf(-at(&T, k + 1, k) * at(&T, k, k + 1));
		inverse_iteration(&H, &x, eigval_re, eigval_im, work, 2);
		x.size[0] = n; x.size[1] = 2;
		matrixf_multiply(&P, &x, &v, 1, 0, 0, 0);
		printf("\n%d) eigval = %0.4f%+.4fi\n   eigvec = \n",
			k + 1, eigval_re, eigval_im);
		for (i = 0; i < n; i++)
			printf("   %9.4f%+.4fi\n", v.data[i], v.data[i + n]);
	}

	free(H.data);
	free(T.data);
	free(P.data);
	free(x.data);

	return 0;
}