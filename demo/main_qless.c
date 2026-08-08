// Compute the solution to the linear system A*x = b without
// explicitly forming the matrix Q (Q-less QR decomposition):
// A'*A*x = A'*b --> R'*Q'*Q*R*x = A'*b --> R'*R*x = A'*b.

#include <stdio.h>
#include "detectum.h"

int main()
{
	float A_data[] = {
		0.358539f, -0.432372f, -0.494633f,
		0.163691f, -0.220791f, -0.214771f,
		-0.017439f, 0.025393f,  0.023145f,
		-0.190184f, 0.225201f,  0.264294f,
		-0.195261f, 0.216867f,  0.275309f
	};
	float b_data[] = { 1, 1, 1, 1, 1 };
	Matrixf A, b;
	Matrixf(x, 3, 1);

	matrixf_init(&A, 5, 3, A_data, 1);
	matrixf_init(&b, 5, 1, b_data, 1);
	printf("\nA = \n"); matrixf_print(&A, "%9.6f ");
	printf("\nb = \n"); matrixf_print(&b, "%9.4f ");
	matrixf_multiply(&A, &b, &x, 1, 0, 1, 0);
	matrixf_decomp_qr(&A, 0, 0, 0);
	matrixf_transpose(&A); matrixf_solve_tril(&A, &x, &b, 0);
	matrixf_transpose(&A); matrixf_solve_triu(&A, &b, &x, 0);
	printf("\nx = \n"); matrixf_print(&x, "%9.4f ");
	return 0;
}