#include <stdio.h>
#include "detectum.h"

#define m 5
#define n 7
#define IN_PLACE

int main()
{
	int i, j;
	float A_data[m * n] = {
		 3.7688e-01f, -6.5364e-02f,  4.3145e-01f,  6.0063e-04f,  3.3769e-01f,  3.8228e-01f, -4.6315e-01f,
		 1.0081e-01f,  1.9233e-02f,  7.4142e-03f,  1.0015e-02f, -6.3969e-03f,  2.4442e-02f, -5.0215e-02f,
		-2.7801e-01f, -7.1722e-03f, -1.4827e-01f, -3.4716e-02f, -9.9950e-02f, -1.6848e-01f,  1.9528e-01f,
		 5.3574e-02f, -1.3641e-02f,  7.8707e-02f,  8.9973e-04f,  6.4819e-02f,  6.5851e-02f, -7.5356e-02f,
		 4.6850e-02f,  2.4224e-02f, -6.0995e-02f,  1.4679e-03f, -6.1765e-02f, -2.8510e-02f,  5.6545e-03f
	};
	float work[n] = { 0 };
	Matrixf A;
#ifdef IN_PLACE
	float data[(m > n ? m : n) * 2];
	Matrixf B = { m, 2, data };
	Matrixf X = { n, 2, data };
#else
	Matrixf(B, m, 2);
	Matrixf(X, n, 2);
#endif

	matrixf_init(&A, m, n, A_data, 1);
	for (j = 0; j < B.cols; j++)
		for (i = 0; i < B.rows; i++)
			at(&B, i, j) = j + 1.0f;
	printf("\nA = \n"); matrixf_print(&A, "%11.7f ");
	printf("\nB = \n"); matrixf_print(&B, "%11.4f ");
	matrixf_solve_qrp(&A, &B, &X, -1, work);
	printf("\nX = \n"); matrixf_print(&X, "%11.4f ");
	return 0;
}