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
#if 1
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
#else // ill-conditioned system
	float C_data[m * n] = {
		-0.1888118f,   0.1218838f,  -0.0250741f,  -0.2557770f,  -0.0106182f,
		-0.1498855f,   0.0969568f,  -0.0197833f,  -0.2031388f,  -0.0083864f,
		-0.0510612f,   0.0331979f,  -0.0067950f,  -0.0690812f,  -0.0030656f,
		 0.1039743f,  -0.0672337f,   0.0138264f,   0.1408075f,   0.0057524f,
		-0.2490493f,   0.1611913f,  -0.0330822f,  -0.3372402f,  -0.0137419f,
		-0.0761835f,   0.0492642f,  -0.0100629f,  -0.1031382f,  -0.0042445f,
		-0.0074878f,   0.0050558f,  -0.0010565f,  -0.0105759f,  -0.0004940f,
		-0.0768361f,   0.0498101f,  -0.0101172f,  -0.1043624f,  -0.0042004f,
		 0.3168127f,  -0.2049339f,   0.0418013f,   0.4291077f,   0.0175003f,
		-0.2445333f,   0.1581392f,  -0.0320945f,  -0.3311207f,  -0.0136784f
	};
	float d_data[m] = {
		 0.0195439f,
		-0.5758212f,
		-0.8009294f,
		-0.3063577f,
		 1.7638617f,
		 0.1355020f,
		-0.4783646f,
		-1.2599086f,
		 0.6109780f,
		-0.5997567f
	};
#endif
	float lb[n], ub[n], work[(m + 2) * n + (m > n ? m : n)];
	Matrixf C, d;
	Matrixf(x, n, 1);
	int j, exitflag;

	matrixf_init(&C, m, n, C_data, 1);
	matrixf_init(&d, m, 1, d_data, 1);
	printf("\nC = \n"); matrixf_print(&C, " %12.6f");
	printf("\nd = \n"); matrixf_print(&d, " %12.6f");
	// Bounded-variable solution
	for (j = 0; j < n; j++) {
		x.data[j] = 0;
		lb[j] = -0.1f;
		ub[j] = +0.1f;
	}
	exitflag = matrixf_solve_bvls(&C, &d, &x, lb, ub, -1, work);
	printf("\nBounded-variable solution");
	printf("\nx = \n"); matrixf_print(&x, " %12.6f");
	printf("\n exitflag = %d\n", exitflag);
	// Nonnegative solution
	for (j = 0; j < n; j++) {
		x.data[j] = 0;
		lb[j] = 0;
		ub[j] = INFINITY;
	}
	exitflag = matrixf_solve_bvls(&C, &d, &x, lb, ub, -1, work);
	printf("\nNonnegative solution");
	printf("\nx = \n"); matrixf_print(&x, " %12.6f");
	printf("\n exitflag = %d\n", exitflag);
	// Unconstrained solution
	matrixf_solve_qr(&C, &d, &x);
	printf("\nUnconstrained solution");
	printf("\nx = \n"); matrixf_print(&x, " %12.6f");
	return 0;
}

#endif