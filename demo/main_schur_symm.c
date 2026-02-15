#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

#define TEST 3

float A_data[] = {
#if (TEST == 0)
	3, 2, 4,
	2, 0, 2,
	4, 2, 3
#elif (TEST == 1)
	- 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	 1, 2,-3, 3, 3, 3, 3, 3, 3, 3,
	 1, 2, 3, 4, 4, 4, 4, 4, 4, 4,
	 1, 2, 3, 4,-5, 5, 5, 5, 5, 5,
	 1, 2, 3, 4, 5, 6, 6, 6, 6, 6,
	 1, 2, 3, 4, 5, 6,-7, 7, 7, 7,
	 1, 2, 3, 4, 5, 6, 7, 8, 8, 8,
	 1, 2, 3, 4, 5, 6, 7, 8,-9, 9,
	 1, 2, 3, 4, 5, 6, 7, 8, 9,10
#elif(TEST == 2)
	- 1, -1,  0,
	-1, -1,  0,
	 0,  0,  0
#elif(TEST == 3)
	- 1,  7,  8,  9, 10, 11,
	 7, +2, 12, 13, 14, 15,
	 8, 12, -3, 16, 17, 18,
	 9, 13, 16, +4, 19, 20,
	10, 14, 17, 19, -5, 21,
	11, 15, 18, 20, 21, +6,
#elif(TEST == 4)
   0.8147237f, 0.9095839f, 0.2027425f,
   0.9095839f, 0.6323593f, 0.3222110f,
   0.2027425f, 0.3222110f, 0.9575068f
#else
	 0, 0, 0, 0, 0, -1e5,
	 0, 0, 0, 0, -1e5, 0,
	 0, 0, 0, -1e5, 0, 0,
	 0, 0, -1e5, 0, 0, 0,
	 0, -1e5, 0, 0, 0, 0,
	 -1e5, 0, 0, 0, 0, 0
#endif
};

int main()
{
	const int n = (int)roundf(sqrtf(sizeof(A_data) / sizeof(A_data[0])));
	Matrixf A = { n, n, A_data };
	Matrixf U = matrixf(n, n);
	Matrixf B = matrixf(n, n);

	printf("A = \n"); matrixf_print(&A, "%9.4g ");
	matrixf_decomp_schur_symm(&A, &U);
	printf("\nD = \n"); matrixf_print(&A, "%9.4g ");
	printf("\nU = \n"); matrixf_print(&U, "%9.4g ");

	matrixf_multiply(&U, &U, &B, 1, 0, 1, 0);
	printf("\nU'*U = \n"); matrixf_print(&B, "%9.4f ");
	matrixf_multiply(&U, &U, &B, 1, 0, 0, 1);
	printf("\nU*U' = \n"); matrixf_print(&B, "%9.4f ");

	matrixf_multiply_inplace(&A, &U, &U, 0, 1, B.data);
	printf("\nU*D*U' = \n"); matrixf_print(&A, "%9.4g ");

	free(U.data);
	free(B.data);

	return 0;
}