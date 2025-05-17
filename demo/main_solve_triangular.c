#include <stdio.h>
#include "detego.h"

#define CASE 1

int main()
{
#if CASE == 1
#define m 5
#define n 3
	float A_data[m * n] = {
		1, 3, -1,
		0, -4, 10,
		0, 0, -2,
		0, 0, 0,
		0, 0, 0
#elif CASE == 2
#define m 3
#define n 5
	float A_data[m * n] = {
	1, 3, -1, 1, 2,
	0, -4, 10, 3, 4,
	0, 0, -2, 5, 6
#elif CASE == 3
#define m 5
#define n 3
	float A_data[m * n] = {
	1, 0, 0,
	3, -4, 0,
	-1, 10, -2,
	1, 2, 3,
	4, 5, 6
#elif CASE == 4
#define m 3
#define n 5
	float A_data[m * n] = {
		1, 0, 0, 0, 0,
		3, -4, 0, 0, 0,
		-1, 10, -2, 0, 0
#endif
	};
	float data[10] = { 0 };
	int i, j;
	Matrixf A, x, b;

	matrixf_init(&A, m, n, A_data, 1);
	matrixf_init(&b, m, 2, data, 0);
	matrixf_init(&x, n, 2, data, 0);
	for (j = 0; j < b.size[1]; j++)
		for (i = 0; i < b.size[0]; i++)
			at(&b, i, j) = (float)j + 1;

	printf("A = [\n"); matrixf_print(&A, "%9.4f "); printf("];\n");
	printf("b = [\n"); matrixf_print(&b, "%9.4f "); printf("];\n");
#if CASE <= 2
	matrixf_solve_triu(&A, &b, &x, 0);
#else
	matrixf_solve_tril(&A, &b, &x, 0);
#endif
	printf("x = [\n"); matrixf_print(&x, "%9.4f "); printf("];\n");

	return 0;
}