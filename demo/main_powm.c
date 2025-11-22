#include <stdio.h>
#include "detectum.h"

static int powm(Matrixf* A, float p, float* work)
{
	const int numel = A->rows * A->cols;
	int i, exitflag = matrixf_log(A, work);

	if (exitflag) {
		return exitflag;
	}
	else {
		for (i = 0; i < numel; i++) {
			A->data[i] *= p;
		}
		return matrixf_exp(A, work);
	}
}

#define n 10

int main()
{
	float A_data[n * n] = {
		7,  2,  1,  5,  7,  8, -3, -4,   3,  1,
		0, -2, -4,  0,  6, -9,  7,  3,  -1, -2,
	   -1, -4,  4, 11,  8, -1, 12, -2,   4,  1,
		1,  4, -4, -6,  1,  4,  5, -3,  -9, -5,
		3, -4, -7,  6,  0,  4,  4,  4,  -8, 10,
	   -7, -2,-10, -9, -5,  4,  4, 10,  -8,  7,
		7,  1,  6,  6, -4, -1,  1, 10,   6,  9,
	   -8, -4, -5,  2, -8, -9, -6,  1,  -7, -7,
	   -8, 12,  5,  4, -7,  0,  9, -1,   1, -7,
	  -10,  7,  5, -9,  3, -7, -2,  8, -11, -2
	};
	float p = -0.6f;
	float work[3 * n * n + n] = { 0 };
	int exitflag;
	Matrixf A;

	matrixf_init(&A, n, n, A_data, 1);
	printf("A = \n"); matrixf_print(&A, "%5.4g "); printf("\n");
	exitflag = powm(&A, p, work);
	printf("A^(%g) = \n", p); matrixf_print(&A, "%9.4f "); printf("\n");
	return exitflag;
}