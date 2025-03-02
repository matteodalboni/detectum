#define DETEGO_USE_PRINT
#include "detego.h"

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
	printf("A = \n"); matrixf_print(&A, "%9.4f ");
	matrixf_exp(&A, work);
	printf("exp(A) = \n"); matrixf_print(&A, "%9.4f ");

	return 0;
}