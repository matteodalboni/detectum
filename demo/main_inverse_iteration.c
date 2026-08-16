#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

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