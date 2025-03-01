#include "detego.h"
#include "detego_utils.h"

#define M 5
#define N 4
#define P 2

int main()
{
	float A_data[M * N] = {
		17, 24, 1, 8,
		23, 5, 7, 14,
		4, 6, 13, 20,
		10, 12, 19, 21,
		11, 18, 25, 2
	};
	float B_data[M * P] = {
		15, 3 * 15,
		16, 3 * 16,
		22, 3 * 22,
		 3, 3 * 3,
		 9, 3 * 9
	};
	Matrixf A = { 0 }, B = { 0 };

	matrixf_init(&A, M, N, A_data, 1);
	matrixf_init(&B, M, P, B_data, 1);
	printf("A = \n"); PRINT("%9.4f ", &A);
	printf("B = \n"); PRINT("%9.4f ", &B);
	matrixf_solve_lsq(&A, &B);
	printf("The (least-squares) solution of the linear system A*X = B is:\n\n");
	printf("X = \n"); PRINT("%9.4f ", &B);

	return 0;
}