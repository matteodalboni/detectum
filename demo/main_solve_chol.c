#include "verto.h"
#include "verto_utils.h"

#define n 6

int main()
{
	float A_data[n * n] = {
		 92,   85,   -4, -60,  -40,  55,
		 85,  147,   44, -84, -111,  97,
		 -4,   44,  152,  86, -125,  23,
		-60,  -84,   86, 334,   -4, -99,
		-40, -111, -125,  -4,  173, -93,
		 55,   97,   23, -99,  -93,  93
	};
	float B_data[n] = {
		128, 178, 176, 173, -200, 76
	};
	float C_data[n * n] = { 0 };
	int i, j;
	Matrixf A, B, C = { {n, n}, C_data };

	matrixf_init(&A, n, n, A_data, 1);
	matrixf_init(&B, n, 1, B_data, 1);
	DISP("%9.4f ", A);
	if (matrixf_solve_spd(&A, &B)) {
		printf("The matrix is not positive definite!\n");
		return 1;
	}
	for (j = 0; j < n; j++)
		for (i = j + 1; i < n; i++) _(&A, i, j) = 0;
	printf("R = \n"); PRINT("%9.4f ", &A);
	DISP("%9.4f ", B);
	matrixf_multiply(&A, &A, &C, 1, 0, 1, 0);
	printf("C = \n"); PRINT("%9.4f ", &C);

	return 0;
}