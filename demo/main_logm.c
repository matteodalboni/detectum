#define DETECTUM_ENABLE_PRINT
#include "detectum.h"

int main()
{
	size_t i;
	float A_data_copy[9] = { 0 };
	float work[30];
	Matrixf(A, 3, 3);
	at(&A, 0, 0) = 1;
	at(&A, 0, 1) = 1;
	at(&A, 1, 2) = 2;
	at(&A, 2, 2) = -1;

	for (i = 0; i < 9; i++) A_data_copy[i] = A_data[i];
	printf("\nA = \n"); matrixf_print(&A, "%9.4f ");
	matrixf_exp(&A, work);
	printf("\nexp(A) = \n"); matrixf_print(&A, "%9.4f ");
	matrixf_log(&A, work);
	printf("\nlog(exp(A)) = \n"); matrixf_print(&A, "%9.4f ");
	for (i = 0; i < 9; i++) A_data_copy[i] -= A_data[i];
	printf("\n||log(exp(A)) - A||_F = %g\n", normf(A_data_copy, 9, 1));

	return 0;
}