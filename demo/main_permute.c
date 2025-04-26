#include <stdio.h>
#include "detego.h"

#define CASE 4

int main()
{
	float A_data[] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16
	};
	float P_data[16] = {
		0, 1, 0, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 0, 1, 0
	};
	float B_data[16] = { 0 };
	float p_data[4] = { 0 };
	int i, j;
	Matrixf A, P, p, B;
	matrixf_init(&A, 4, 4, A_data, 1);
	matrixf_init(&B, 4, 4, B_data, 1);
	matrixf_init(&P, 4, 4, P_data, 1);
	for (j = 0; j < 4; j++) {
		i = 0;
		while (at(&P, i, j) == 0) {
			i++;
		}
		p_data[j] = (float)i;
	}
	printf("A = \n"); matrixf_print(&A, "%5.0f "); printf("\n");
	printf("P = \n"); matrixf_print(&P, "%5.0f "); printf("\n");
#if CASE == 1 // p encodes a column permutation and A <-- A*P
	printf("A*P = \n");
	matrixf_init(&p, 1, 4, p_data, 0);
	matrixf_multiply(&A, &P, &B, 1, 0, 0, 0);
	matrixf_permute(&A, &p, 0);
#elif CASE == 2 // p encodes a column permutation and A <-- P*A
	printf("P*A = \n");
	matrixf_init(&p, 1, 4, p_data, 0);
	matrixf_multiply(&P, &A, &B, 1, 0, 0, 0);
	matrixf_permute(&A, &p, 1);
#elif CASE == 3 // p encodes a row permutation and A <-- P'*A
	printf("P'*A = \n");
	matrixf_init(&p, 4, 1, p_data, 0);
	matrixf_multiply(&P, &A, &B, 1, 0, 1, 0);
	matrixf_permute(&A, &p, 0);
#else // p encodes a row permutation and A <-- A*P'
	printf("A*P' = \n");
	matrixf_init(&p, 4, 1, p_data, 0);
	matrixf_multiply(&A, &P, &B, 1, 0, 0, 1);
	matrixf_permute(&A, &p, 1);
#endif
	for (i = 0; i < 16; i++) {
		if (A.data[i] != B.data[i]) {
			return -1;
		}
	}
	matrixf_print(&A, "%5.0f ");
	printf("\np = \n"); matrixf_print(&p, "%5.0f ");

	return 0;
}