#include "verto.h"
#include "verto_utils.h"

#define N 3

int main()
{
	float A_data[] = {
		1, 1, 0, 
		0, 0, 2, 
		0, 0, -1
	};
	float work[N * (3 * N + 1)] = { 0 };
	Matrixf A = { 0 };

	matrixf_init(&A, N, N, A_data, 1);
	printf("A = \n"); PRINT("%9.4f ", &A);
	matrixf_exp(&A, work);
	printf("exp(A) = \n"); PRINT("%9.4f ", &A);

	return 0;
}