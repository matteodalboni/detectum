#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

#if 0

#define m 1000
#define n 900

int main()
{
	int j, exitflag;
	float* lb = malloc(sizeof(float) * n);
	float* ub = malloc(sizeof(float) * n);
	float* work = malloc(sizeof(float) * ((m + 2) * n + (m > n ? m : n)));
	FILE* C_file = fopen("../C.bin", "rb");
	FILE* d_file = fopen("../d.bin", "rb");
	FILE* x_file = fopen("../x.bin", "wb");
	Matrixf C = matrixf(m, n);
	Matrixf d = matrixf(m, 1);
	Matrixf x = matrixf(n, 1);

	if (!lb || !ub) return -10;
	fread(C.data, sizeof(float), m * n, C_file); fclose(C_file);
	fread(d.data, sizeof(float), m * 1, d_file); fclose(d_file);
	for (j = 0; j < n; j++) {
		x.data[j] = 0;
		lb[j] = -0.1f;
		ub[j] = +0.1f;
	}
	exitflag = matrixf_solve_bvls(&C, &d, &x, lb, ub, -1, work);
	printf("\n exitflag = %d\n", exitflag);
	fwrite(x.data, sizeof(float), n, x_file); fclose(x_file);
	free(lb);
	free(ub);
	free(work);
	free(C.data);
	free(d.data);
	free(x.data);
	return 0;
}

#else

#define m 10
#define n 5

int main()
{
	float C_data[m * n] = {
		92, 99,   1,  8, 15,
		98, 80,   7, 14, 16,
		 4, 81,  88, 20, 22,
		85, 87,  19, 21,  3,
		86, 93,  25,  2,  9,
		17, 24,  76, 83, 90,
		23,  5,  82, 89, 91,
		79,  6,  13, 95, 97,
		10, 12,  94, 96, 78,
		11, 18, 100, 77, 84
	};
	float d_data[m] = { 10, -10, 10, 10, 10, -10, 10, -10, 10, 10 };
	float lb[n], ub[n], work[(m + 2) * n + (m > n ? m : n)];
	Matrixf C, d;
	Matrixf(x, n, 1);
	int j, exitflag;

	matrixf_init(&C, m, n, C_data, 1);
	matrixf_init(&d, m, 1, d_data, 1);
	printf("\nC = \n"); matrixf_print(&C, " %9.4f");
	printf("\nd = \n"); matrixf_print(&d, " %9.4f");
	// Bounded-variable solution
	for (j = 0; j < n; j++) {
		lb[j] = -0.1f;
		ub[j] = +0.1f;
	}
	exitflag = matrixf_solve_bvls(&C, &d, &x, lb, ub, -1, work);
	printf("\nBounded-variable solution");
	printf("\nx = \n"); matrixf_print(&x, " %9.4f");
	printf("\n exitflag = %d\n", exitflag);
	// Nonnegative solution
	for (j = 0; j < n; j++) {
		lb[j] = 0;
		ub[j] = INFINITY;
	}
	exitflag = matrixf_solve_bvls(&C, &d, &x, lb, ub, -1, work);
	printf("\nNonnegative solution");
	printf("\nx = \n"); matrixf_print(&x, " %9.4f");
	printf("\n exitflag = %d\n", exitflag);
	// Unconstrained solution
	matrixf_solve_qr(&C, &d, &x);
	printf("\nUnconstrained solution");
	printf("\nx = \n"); matrixf_print(&x, " %9.4f");
	return 0;
}

#endif