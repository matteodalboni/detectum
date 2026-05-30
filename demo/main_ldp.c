// Solve for x the LDP problem
//  minimize   x'x 
//    s.t.   Ax <= b
// by nonpositive least squares.
// See Lawson, Charles L., and Richard J. Hanson. 
// "Solving least squares problems". Society for 
// Industrial and Applied Mathematics, 1995.

#include <stdio.h>
#include "detectum.h"

#define m 3 // number of constraints
#define n 4 // number of variables

#define WORKLEN(m,n) (((m)+2)*(n)+((m)>(n)?(m):(n)))

int main()
{
	float C_data[m * (n + 1)] = { // C = [A, b]
		-0.2027f, -0.2721f, -0.7467f, -0.4659f,	-0.5251f,
		-0.1987f, -0.1988f, -0.4450f, -0.4186f,	-0.2026f,
		-0.6037f, -0.0152f, -0.9318f, -0.8462f,	-0.6721f
	};
	float lb[m], ub[m], nrm;
	float work[WORKLEN(n + 1, m)];
	int i;
	Matrixf(d, n + 1, 1);
	Matrixf(y, m, 1);
	Matrixf C;

	matrixf_init(&C, m, n + 1, C_data, 1);
	for (nrm = 0.0f, i = 0; i < m; i++) {
		if (fabsf(at(&C, i, n)) > nrm) {
			nrm = fabsf(at(&C, i, n));
		}
		lb[i] = -INFINITY;
		ub[i] = 0.0f;
	}
	for (i = 0; i < n; i++) {
		at(&d, i, 0) = 0.0f;
	}
	at(&d, n, 0) = nrm;
	matrixf_transpose(&C);
	matrixf_solve_bvls(&C, &d, &y, lb, ub, -1, work);
	matrixf_multiply(&C, &y, &d, -1.0f, 1.0f, 0, 0);
	if (at(&d, n, 0) != 0.0f) {
		for (i = 0; i < n; i++) {
			at(&d, i, 0) /= -at(&d, n, 0);
		}
	}
	d.rows = n;
	printf("x = \n"); matrixf_print(&d, "%9.4f");
	return 0;
}