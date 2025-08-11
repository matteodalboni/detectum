#include <stdio.h>
#include "detectum.h"

int main()
{
	float work[30];
	Matrixf_(A, 3, 3);
	at(&A, 0, 0) = 1;
	at(&A, 0, 1) = 1;
	at(&A, 0, 2) = 0;
	at(&A, 1, 0) = 0;
	at(&A, 1, 1) = 0;
	at(&A, 1, 2) = 2;
	at(&A, 2, 0) = 0;
	at(&A, 2, 1) = 0;
	at(&A, 2, 2) = -1;

	printf("\nA = \n"); matrixf_print(&A, "%9.4f ");
	matrixf_exp(&A, work);
	printf("\nexp(A) = \n"); matrixf_print(&A, "%9.4f ");
	matrixf_log(&A, work);
	printf("\nlog(exp(A)) = \n"); matrixf_print(&A, "%9.4f ");

	return 0;
}