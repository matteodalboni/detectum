#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

// This function computes the right eigenvector v of a square matrix A corresponding
// to the eigenvalue eigval_re+eigval_im*i using the inverse iteration method. 
// Typically, if A is not defective, 2-3 iterations produce a good approximation. 
// The input vector v must be initialized to a nonzero vector to enable convergence.
// If eigval_im is zero, v must be an n-by-1 or n-by-2 matrix, where n is the number
// of rows of A. If eigval_im is nonzero, v must be an n-by-2 matrix. On output, the
// first column of v contains the real parts, and, if present, the second column 
// contains the imaginary parts. Also, on output, matrix A remains unchanged.
// iter specifies the number of iterations to perform.
// The array work is the additional workspace memory: if eigval_im is zero, its 
// minimum length is n*n+n; otherwise, its minimum length is 4*n*n+2*n. 
// On size mismatch or non-square matrix, the function returns -1. On success, it 
// returns 0.
int matrixf_get_eigenvector(Matrixf* A, Matrixf* v,
	float eigval_re, float eigval_im, int iter, float* work)
{
	int i, j;
	const size_t n = A->rows;
	const size_t q = v->cols;
	const int p = eigval_im == 0 ? 1 : 2;
	float nrm_inv, nrmA = normf(A->data, n * n, 1);
	float del = nrmA * DETECTUM_FLT_MIN;
	float tol = nrmA * DETECTUM_FLT_EPS;
	Matrixf perm = { p * n, 1, work };
	Matrixf C = { p * n, p * n, work + p * n };

	if (A->cols != n || q > 2 ||
		v->rows != n || q < p) {
		return -1;
	}
	eigval_re += epsf(eigval_re);
	eigval_im += epsf(eigval_im);
	if (fabsf(eigval_re) < tol) {
		eigval_re = eigval_re < 0 ? -tol : tol;
	}
	for (j = 0; j < n; j++) {
		for (i = 0; i < n; i++) {
			at(&C, i, j) = at(A, i, j);
			if (at(&C, i, j) == 0) {
				at(&C, i, j) = del;
			}
			if (p == 2) {
				at(&C, i + n, j + n) = at(A, i, j);
				at(&C, i + n, j) = 0;
				at(&C, i, j + n) = 0;
			}
		}
		at(&C, j, j) -= eigval_re;
		if (p == 2) {
			at(&C, j + n, j + n) -= eigval_re;
			at(&C, j + n, j) = -eigval_im;
			at(&C, j, j + n) = +eigval_im;
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
		nrm_inv = 1.0f / normf(v->data, p * n, 1);
		for (j = 0; j < p * n; j++) {
			v->data[j] *= nrm_inv;
		}
	}
	v->rows = n;
	v->cols = q;
	return 0;
}

#define TEST 0
#define ITER 5

#if TEST == 1 // defective matrix
#define n 8
float A_data[] = {
	3, 1, 0, 0, 0, 0, 0, 0,
	0, 3, 1, 0, 0, 0, 0, 0,
	0, 0, 3, 1, 0, 0, 0, 0,
	0, 0, 0, 3, 1, 0, 0, 0,
	0, 0, 0, 0, 3, 1, 0, 0,
	0, 0, 0, 0, 0, 3, 1, 0,
	0, 0, 0, 0, 0, 0, 3, 1,
	0, 0, 0, 0, 0, 0, 0, 3
};
#elif TEST == 2
#define n 8
float A_data[] = {
	0, -1,  1,  0,  0,  0,  0,  0,
	1,  0,  0,  1,  0,  0,  0,  0,
	0,  0,  0, -1,  0,  0,  0,  0,
	0,  0,  1,  0,  0,  0,  0,  0,
	0,  0,  0,  0,  0, -1,  1,  0,
	0,  0,  0,  0,  1,  0,  0,  1,
	0,  0,  0,  0,  0,  0,  0, -1,
	0,  0,  0,  0,  0,  0,  1,  0
};
#elif TEST == 3
#define n 3
float A_data[] = {
	 1,  1,  1,
	 1,  1,  1,
	-1, -1, -1
};
#elif TEST == 4
#define n 3
float A_data[] = {
	 1,  1,  1,
	 0,  1,  0,
	 0,  0,  1
};
#else
#define n 4
float A_data[] = {
	 1, 1, 1, 3,
	 1, 2, 1, 1,
	 1, 1, 3, 1,
	-2, 1, 1, 4
};
#endif

int main()
{
	int i, k = 0;
	float eigval_re = 0, eigval_im = 0;
	float work[4 * n * n + 2 * n] = { 0 };
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
			v.data[i] = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
		}
		matrixf_get_eigenvector(&A, &v, eigval_re, eigval_im, ITER, work);
		printf("\neigval(:,%d) = %0.4f%+.4fi;\neigvec(:,%d) = [\n",
			k + 1, eigval_re, eigval_im, k + 1);
		for (i = 0; i < n; i++)
			printf("   %9.4f%+.4fi\n", v.data[i], v.data[i + n]);
		printf("];\n");
	}
	return 0;
}