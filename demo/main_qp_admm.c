// Solve for x the QP problem
//  minimize   1/2 x'Qx + c'x 
//    s.t.     rl <= Ax <= ru
// by regularized Alternating Direction Method of Multipliers (ADMM).
// See B. Stellato, G. Banjac, P. Goulart, A. Bemporad and S. Boyd, 
// "OSQP: An Operator Splitting Solver for Quadratic Programs," 2018
// UKACC 12th International Conference on Control (CONTROL), 
// Sheffield, UK, 2018, pp. 339-339, doi: 10.1109/CONTROL.2018.8516834.

#include <stdio.h>
#include "detectum.h"

#define PROBLEM (2)

#if (PROBLEM == 0)
#define m 3 // number of constraints
#define n 2 // number of variables
float Q_data[n * n] = { 4, 1, 1, 2 };
float c_data[n] = { 1, 1 };
float A_data[m * n] = { 1, 1, 1, 0, 0, 1 };
float rl[m] = { 1, 0, 0 };
float ru[m] = { 1, 0.7f, 0.7f };

#elif (PROBLEM == 1)
#define m 3 // number of constraints
#define n 2 // number of variables
float Q_data[n * n] = { 1, -1, -1, 2 };
float c_data[n] = { -2, -6 };
float A_data[m * n] = { 1, 1, -1, 2, 2, 1 };
float rl[m] = { -1e38f, -1e38f, -1e38f };
float ru[m] = { 2, 2, 3 };

#elif (PROBLEM == 2)
#define m 5 // number of constraints
#define n 5 // number of variables
float Q_data[n * n] = {
	39985, 33980, 11140, 16915, 17055,
	33980, 40085, 17005, 11090, 11290,
	11140, 17005, 40285, 33880, 33640,
	16915, 11090, 33880, 40085, 39605,
	17055, 11290, 33640, 39605, 39985
};
float c_data[n] = {
	-1170,
	-2850,
	-3130,
	-1210,
	-990
};
float A_data[m * n] = {
	1, 0, 0, 0, 0,
	0, 1, 0, 0, 0,
	0, 0, 1, 0, 0,
	0, 0, 0, 1, 0,
	0, 0, 0, 0, 1
};
float rl[m] = { -0.1f, -0.1f, -0.1f, -0.1f, -0.1f };
float ru[m] = { +0.1f, +0.1f, +0.1f, +0.1f, +0.1f };
#endif

int main()
{
	int iter = 100;
	float sigma = 1e-4f;
	float rho = sqrtf(normf(Q_data, n * n, 1));
	int i, j;
	float g, h;
	Matrixf Q, c, A;
	Matrixf(x, n, 1);
	Matrixf(s, m, 1);
	Matrixf(v, m, 1);

	// Initialize
	matrixf_init(&Q, n, n, Q_data, 1);
	matrixf_init(&c, n, 1, c_data, 0);
	matrixf_init(&A, m, n, A_data, 1);
	matrixf_multiply(&A, &A, &Q, rho, 1.0f, 1, 0);
	for (j = 0; j < n; j++) {
		at(&Q, j, j) += sigma;
	}
	matrixf_decomp_qr(&Q, 0, 0, 0);
	// Loop
	while (iter--) {
		for (j = 0; j < n; j++) {
			x.data[j] = sigma * x.data[j] - c.data[j];
		}
		for (i = 0; i < m; i++) {
			s.data[i] -= v.data[i];
		}
		matrixf_multiply(&A, &s, &x, rho, 1.0f, 1, 0);
		matrixf_unpack_house(&Q, &x, 0, 1);
		matrixf_solve_triu(&Q, &x, &x, 0);
		matrixf_multiply(&A, &x, &s, 1.0f, 0.0f, 0, 0);
		for (i = 0; i < m; i++) {
			h = g = s.data[i];
			g += v.data[i];
			g = g < rl[i] ? rl[i] : (g > ru[i] ? ru[i] : g);
			v.data[i] += h - g;
			s.data[i] = g;
		}
	}
	// Print solution
	printf("\nx = \n"); matrixf_print(&x, " % 12.6f");
	return 0;
}