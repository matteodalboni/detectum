// Solve for x the convex QP problem
//  minimize   1/2 x'Qx + c'x 
//    s.t.        Ax <= b
// by nonnegative least squares.
// See A. Bemporad, "A Quadratic Programming Algorithm Based on
// Nonnegative Least Squares With Applications to Embedded Model
// Predictive Control," in IEEE Transactions on Automatic 
// Control, vol. 61, no. 4, pp. 1111-1116, April 2016, 
// doi: 10.1109/TAC.2015.2459211.

#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

#define m 3 // number of constraints
#define n 4 // number of variables

#define WORKLEN(m,n) (((m)+2)*(n)+((m)>(n)?(m):(n)))

int main()
{
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
		0.2027f, 0.2721f, 0.7467f, 0.4659f,
		0.1987f, 0.1988f, 0.4450f, 0.4186f,
		0.6037f, 0.0152f, 0.9318f, 0.8462f
	};
	float b_data[m] = {
		0.5251f,
		0.2026f,
		0.6721f
	};
	float* lb = malloc(sizeof(float) * m);
	float* ub = malloc(sizeof(float) * m);
	float* work = malloc(sizeof(float) * WORKLEN(n + 1, m));
	float nrm, alpha;
	int i, j, info;
	Matrixf Q, c, A, b;
	Matrixf C = matrixf(n + 1, m);
	Matrixf d = matrixf(n + 1, 1);
	Matrixf y = matrixf(m, 1);

	// Initialize
	matrixf_init(&Q, n, n, Q_data, 1);
	matrixf_init(&c, n, 1, c_data, 0);
	matrixf_init(&A, m, n, A_data, 1);
	matrixf_init(&b, m, 1, b_data, 0);
	// Assemble matrices for NNLS solver
	matrixf_solve_chol(&Q, &c);
	matrixf_multiply(&A, &c, &b, 1.0f, 1.0f, 0, 0);
	matrixf_transpose(&A);
	matrixf_transpose(&Q);
	matrixf_solve_tril(&Q, &A, &A, 0);
	matrixf_transpose(&Q);
	for (nrm = 0.0f, j = 0; j < m; j++) {
		for (i = 0; i < n; i++) {
			at(&C, i, j) = -at(&A, i, j);
		}
		at(&C, n, j) = -at(&b, j, 0);
		if (fabsf(at(&b, j, 0)) > nrm) {
			nrm = fabsf(at(&b, j, 0));
		}
		lb[j] = 0.0f;
		ub[j] = INFINITY;
	}
	for (i = 0; i < n; i++) {
		at(&d, i, 0) = 0.0f;
	}
	at(&d, n, 0) = 1.0f + nrm;
	// Solve NNLS problem
	info = matrixf_solve_bvls(&C, &d, &y, lb, ub, -1, work);
	if (info < 0) {
		return info;
	}
	// Assemble QP solution
	d.rows = n;
	matrixf_multiply(&A, &y, &d, 1.0f, 0.0f, 0, 0);
	matrixf_solve_triu(&Q, &d, &d, 0);
	for (alpha = 1.0f + nrm, j = 0; j < m; j++) {
		alpha += at(&b, j, 0) * at(&y, j, 0);
	}
	for (i = 0; i < n; i++) {
		at(&c, i, 0) = -at(&c, i, 0) - at(&d, i, 0) / alpha;
	}
	printf("x = \n"); matrixf_print(&c, "%9.4f");
	// Free memory
	free(lb);
	free(ub);
	free(work);
	free(C.data);
	free(d.data);
	free(y.data);
	return 0;
}