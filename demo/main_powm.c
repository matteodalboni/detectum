#include <stdio.h>
#include "detectum.h"

// This function raises the n-by-n matrix A to the p-th positive integer power.
// The algorithm is based on the binary expansion of p to minimize the number of
// matrix multiplications. The array work is the additional workspace memory:
// its minimum length is 2*n*n. The function returns -1 if the input matrix is 
// not square.
static int mpoweri(Matrixf* A, unsigned const int p, float* work)
{
	int j, k = 0;
	const int n = A->size[0];
	unsigned int s = 1, i = 0;
	Matrixf Z = { { n, n }, work };
	Matrixf F = { { n, n }, work + n * n };

	if (A->size[1] != n) {
		return -1;
	}
	if (p == 0) {
		for (k = 0; k < n * n; k++) {
			A->data[k] = !(k % (n + 1));
		}
	}
	else if (p > 1) {
		for (j = 0; j < n * n; j++) {
			Z.data[j] = A->data[j];
		}
		while (!((p & (s << i)) >> i)) {
			matrixf_multiply(&Z, &Z, A, 1, 0, 0, 0);
			for (j = 0; j < n * n; j++) {
				Z.data[j] = A->data[j];
			}
			i++;
		}
		for (j = 0; j < n * n; j++) {
			F.data[j] = Z.data[j];
		}
		while ((s << ++i) <= p) {
			matrixf_multiply(&Z, &Z, A, 1, 0, 0, 0);
			for (j = 0; j < n * n; j++) {
				Z.data[j] = A->data[j];
			}
			if ((p & (s << i)) >> i) {
				matrixf_multiply(&F, &Z, A, 1, 0, 0, 0);
				for (j = 0; j < n * n; j++) {
					F.data[j] = A->data[j];
				}
			}
		}
	}
	return 0;
}

int main()
{
	float A_data[3 * 3] = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
		0.7f, 0.8f, 0.9f
	};
	float work[2 * 3 * 3] = { 0 };
	Matrixf A;
	unsigned int p = 15;

	matrixf_init(&A, 3, 3, A_data, 1);
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	mpoweri(&A, p, work);
	printf("A^%d = \n", p); matrixf_print(&A, "%9.4f "); printf("\n");

	return 0;
}