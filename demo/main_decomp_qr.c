#include <stdio.h>
#include "detectum.h"

#define TEST 3

int main()
{
#if (TEST == 0)
#define M 3
#define N 3
	float A_data[] = {
	1, 1, 0,
	0, 0, 2,
	0, 0,-1
	};
#elif (TEST == 1)
#define M 4
#define N 2
	float A_data[] = {
	1, 2,
	3, 4,
	5, 6,
	7, 8
	};
#elif (TEST == 2)
#define M 2
#define N 4
	float A_data[] = {
	1, 3, 5, 7,
	2, 4, 6, 8
	};
#elif (TEST == 3)
#define M 10
#define N 5
	float A_data[] = {
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
#else
#define M 1
#define N 2
	float A_data[] = { 1, 2 };
#endif
	int i;
	float Q_data[M * M] = { 0 }, QR_data[M * N] = { 0 }, P_data[N * N] = { 0 };
	Matrixf A = { 0 }, Q = { 0 }, QR = { 0 }, P = { 0 };

	for (i = 0; i < N * N; i++) P_data[i] = !(i % (N + 1));

	matrixf_init(&A, M, N, A_data, 1);
	matrixf_init(&Q, M, M, Q_data, 0);
	matrixf_init(&P, N, N, P_data, 0);
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");

	matrixf_decomp_qr(&A, &Q, &P, 0);
	printf("Q = \n"); matrixf_print(&Q, "%9.4f "); printf("\n");
	printf("R = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	printf("P = \n"); matrixf_print(&P, "%9.4f "); printf("\n");

	matrixf_init(&QR, M, N, QR_data, 0);
	matrixf_multiply(&Q, &A, &QR, 1, 0, 0, 0);
	printf("Q*R = \n"); matrixf_print(&QR, "%9.4f "); printf("\n");
	A.rows = M;
	A.cols = N;
	matrixf_multiply(&QR, &P, &A, 1, 0, 0, 1);
	printf("Q*R*P' = \n"); matrixf_print(&A, "%9.4f "); printf("\n");

	return 0;
}