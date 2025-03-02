#include "detego.h"
#include "detego_utils.h"

#define n 4
#define p 4

int main()
{
	float A_data[n * n] = {
		 4, -1, -6,  0,
		-5, -4, 10,  8,
		 0,  9,  4, -2,
		 1,  0, -7,  5,
	};
	float B_data[n * p] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
		//2, 21, -12, -6
	};
	float P_data[n * n] = { 0 };
	float L_data[n * n] = { 0 };
	Matrixf A, B, P, L;
	int i, j;

	matrixf_init(&A, n, n, A_data, 1); DISP("%9.4f ", A);
	if (1) {
		matrixf_init(&B, n, p, B_data, 1); DISP("%9.4f ", B);
		if (matrixf_solve_lu(&A, &B)) return 1; 
		printf("X = \n"); PRINT("%9.4f ", &B);
	}
	else
	{
		matrixf_init(&P, n, n, P_data, 0);
		matrixf_init(&L, n, n, L_data, 0);
		for (j = 0; j < n; j++)
			for (i = 0; i < n; i++) {
				at(&P, i, j) = (float)(i == j);
				at(&L, i, j) = (float)(i == j);
			}
		if (matrixf_decomp_lu(&A, &P)) return 1;
		for (j = 0; j < n; j++)
			for (i = j + 1; i < n; i++) {
				at(&L, i, j) = at(&A, i, j);
				at(&A, i, j) = 0;
			}
		printf("U = \n"); PRINT("%9.4f ", &A);
		DISP("%9.4f ", L);
		DISP("%9.4f ", P);
	}

	return 0;
}