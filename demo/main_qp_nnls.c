// Solve for x the convex QP problem
//  minimize   1/2 x'*Q*x + c'*x 
//    s.t.     Aineq*x <= bineq
//               Aeq*x  = beq
// by partially nonnegative least squares (PNNLS).
// [1] A. Bemporad, "A Quadratic Programming Algorithm Based on 
// Nonnegative Least Squares With Applications to Embedded Model 
// Predictive Control," in IEEE Transactions on Automatic Control, vol.
// 61, no. 4, pp. 1111-1116, April 2016, doi: 10.1109/TAC.2015.2459211.
// [2] A.Bemporad, "A Numerically Stable Solver for Positive
// Semidefinite Quadratic Programs Based on Nonnegative Least Squares,"
// in IEEE Transactions on Automatic Control, vol. 63, no. 2, pp.
// 525 - 531, Feb. 2018, doi : 10.1109/TAC.2017.2735938.

#include <stdio.h>
#include "detectum.h"

#define m 5 // number of constraints
#define n 4 // number of variables
#define mi 3 // number of inequality constraints
#define WORKLEN(m,n) ((m)*(n)+2*(n)+((m)>(n)?(m):(n)))

float Q_data[n * n] = {
	2.3546f, 1.6361f, 1.8427f, 2.1537f,
	1.6361f, 1.6617f, 1.5320f, 1.4873f,
	1.8427f, 1.5320f, 2.4314f, 2.2958f,
	2.1537f, 1.4873f, 2.2958f, 2.8471f
};
float c_data[n] = {
	-0.7583f,
	-0.2899f,
	-1.0962f,
	-1.2270f
};
float A_data[m * n] = {
	// Inequality constraints:
	0.2027f, 0.2721f, 0.7467f, 0.4659f,
	0.1987f, 0.1988f, 0.4450f, 0.4186f,
	0.6037f, 0.0152f, 0.9318f, 0.8462f,
	// Equality constraints:
	3, 5, 7, 9,
	1, 0, 0, 2
};
float b_data[m] = {
	// Inequality constraints:
	0.5251f,
	0.2026f,
	0.6721f,
	// Equality constraints:
	4,
	0
};

int main()
{
	float lb[m];
	float ub[m];
	float work[WORKLEN(n + 1, m)];
	float C_data[(n + 1) * m];
	float d_data[n + 1];
	float x_data[n] = { 0 };
	float y_data[m] = { 0 };
	float nrm, alpha, tol = -1.0f, epsilon = 10.0f;
	int i, j, info, iter_prox = 200;
	Matrixf Q, c, A, b, C, d, x, y, h;
	FILE* x_file = fopen("../x.bin", "wb");

	// Initialize
	matrixf_init(&Q, n, n, Q_data, 1);
	matrixf_init(&c, n, 1, c_data, 0);
	matrixf_init(&A, m, n, A_data, 1);
	matrixf_init(&b, m, 1, b_data, 0);
	matrixf_init(&x, n, 1, x_data, 0);
	matrixf_init(&y, m, 1, y_data, 0);
	matrixf_init(&C, n + 1, m, C_data, 0);
	matrixf_init(&d, n + 1, 1, d_data, 0);
	matrixf_init(&h, m, 1, C_data + m * n, 0);
	for (j = 0; j < n; j++) {
		at(&Q, j, j) += epsilon;
	}
	while (iter_prox--) { // proximal-point iterations
		// Assemble matrices for PNNLS solver
		for (j = 0; j < n; j++) {
			x.data[j] = c.data[j] - epsilon * x.data[j];
			d.data[j] = 0.0f;
		}
		matrixf_solve_chol(&Q, &x);
		matrixf_transpose(&A);
		matrixf_transpose(&Q);
		C.rows = n;
		matrixf_solve_tril(&Q, &A, &C, 0);
		matrixf_transpose(&A);
		matrixf_transpose(&Q);
		matrixf_transpose(&C);
		C.cols = n + 1;
		for (j = 0; j < m; j++) {
			h.data[j] = b.data[j];
		}
		matrixf_multiply(&A, &x, &h, 1.0f, 1.0f, 0, 0);
		for (nrm = 0.0f, j = 0; j < m; j++) {
			if (fabsf(at(&h, j, 0)) > nrm) {
				nrm = fabsf(at(&h, j, 0));
			}
			lb[j] = j < mi ? 0.0f : -INFINITY;
			ub[j] = INFINITY;
		}
		for (j = 0; j < (n + 1) * m; j++) {
			C.data[j] *= -1.0f;
		}
		matrixf_transpose(&C);
		at(&d, n, 0) = 1.0f + nrm;
		// Solve PNNLS problem
		info = matrixf_solve_bvls(&C, &d, &y, lb, ub, tol, work);
		if (info < 0) {
			return info;
		}
		// Assemble QP solution
		matrixf_transpose(&C);
		d.rows = n;
		C.cols = n;
		matrixf_multiply(&C, &y, &d, -1.0f, 0.0f, 1, 0);
		matrixf_solve_triu(&Q, &d, &d, 0);
		for (alpha = 1.0f + nrm, j = 0; j < m; j++) {
			alpha -= at(&h, j, 0) * at(&y, j, 0);
		}
		for (i = 0; i < n; i++) {
			at(&x, i, 0) = -at(&x, i, 0) - at(&d, i, 0) / alpha;
		}
		d.rows = n + 1;
		C.rows = n + 1;
		C.cols = m;
		fwrite(x.data, sizeof(float), n, x_file);
		// Recreate Q
		for (j = 0; j < n; j++) {
			at(&Q, j, j) *= at(&Q, j, j);
			for (i = j - 1; i >= 0; i--) {
				at(&Q, j, j) += at(&Q, i, j) * at(&Q, i, j);
				at(&Q, i, j) = at(&Q, j, i);
			}
		}
	}
	// Done!
	printf("x = \n"); matrixf_print(&x, "%12.6f");
	fclose(x_file);
	return 0;
}