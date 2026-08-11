#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

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
		matrixf_get_eigenvector(&A, &v, eigval_re, eigval_im, 2, work);
		printf("\neigval(:,%d) = %0.4f%+.4fi;\neigvec(:,%d) = [\n",
			k + 1, eigval_re, eigval_im, k + 1);
		for (i = 0; i < n; i++)
			printf("   %9.4f%+.4fi\n", v.data[i], v.data[i + n]);
		printf("];\n");
	}
	return 0;
}