#include <stdio.h>
#include "detectum.h"

// This function performs the LTL' decomposition with pivoting using the method
// of Aasen. The n-by-n symmetric indefinite matrix A is decomposed so that  
// P*A*P' = L*T*L', where L is unit lower triangular, T is symmetric tridiagonal,
// and P is a permutation matrix. The matrix A is transformed so that:
// 1) its diagonal contains the main diagonal of T;
// 2) its superdiagonal stores the first diagonals of T;
// 3) its strictly lower part holds the strictly lower part of L, assuming also
//    L(i,0) = 0 for 0 < i < n;
// 4) the permutation matrix P is encoded in the first column of A so that 
//    P(0,0) = 1 and P(i,A(i,0)) = 1 for 0 < i < n.
// The function returns -1 if A is not square. On success, it returns 0.
int matrixf_decomp_ltl(Matrixf* A)
{
	int i, j, k, p;
	const int n = A->rows;
	float tmp, P1, A0 = A->data[0], * h = A->data;
	float* col1, * col2;

	if (A->cols != n) {
		return -1;
	}
	for (j = 1; j < n; j++) {
		at(A, 0, j) = (float)j;
	}
	for (j = 0; j < n; j++) {
		if (j > 0) {
			h[0] = at(A, 0, 1);
			if (j > 1) {
				h[0] *= at(A, j, 1);
				if (j == 2) {
					h[1] = at(A, 1, 1) * at(A, 2, 1) + at(A, 1, 2);
				}
				else {
					h[1] = at(A, 1, 1) * at(A, j, 1)
						+ at(A, 1, 2) * at(A, j, 2);
					for (k = 2; k < j - 1; k++) {
						h[k] = at(A, k, k) * at(A, j, k)
							+ at(A, k - 1, k) * at(A, j, k - 1)
							+ at(A, k, k + 1) * at(A, j, k + 1);
					}
					h[j - 1] = at(A, j - 1, j - 1) * at(A, j, j - 1)
						+ at(A, j - 2, j - 1) * at(A, j, j - 2)
						+ at(A, j - 1, j);
				}
			}
			for (tmp = 0, k = 1; k < j; k++) {
				tmp += at(A, j, k) * h[k];
			}
			h[j] = at(A, j, j) - tmp;
			at(A, j, j) = h[j];
			if (j > 1) {
				at(A, j, j) -= at(A, j - 1, j) * at(A, j, j - 1);
			}
			for (i = j + 1; i < n; i++) {
				for (tmp = 0, k = 1; k <= j; k++) {
					tmp += at(A, i, k) * h[k];
				}
				h[i] = at(A, j, i) - tmp;
			}
		}
		if (j < n - 1) {
			for (p = j + 1, k = j + 1; k < n; k++) {
				if (fabsf(h[k]) > fabsf(h[p])) {
					p = k;
				}
			}
			for (k = 0; k < n; k++) {
				tmp = at(A, p, k);
				at(A, p, k) = at(A, j + 1, k);
				at(A, j + 1, k) = tmp;
			}
			col1 = &at(A, 0, p);
			col2 = &at(A, 0, j + 1);
			tmp = col1[0];
			col1[0] = col2[0];
			col2[0] = tmp;
			for (k = j + 1; k < n; k++) {
				tmp = col1[k];
				col1[k] = col2[k];
				col2[k] = tmp;
			}
			if (j == 0) {
				P1 = at(A, 0, 1);
			}
			at(A, j, j + 1) = h[j + 1];
			if (j < n - 2) {
				if (h[p] != 0) {
					for (i = j + 2; i < n; i++) {
						at(A, i, j + 1) = h[i] / h[j + 1];
					}
				}
				else {
					for (i = j + 2; i < n; i++) {
						at(A, i, j + 1) = 0;
					}
				}
			}
		}
	}
	A->data[0] = A0;
	if (n > 1) {
		A->data[1] = P1;
	}
	for (j = 2; j < n; j++) {
		at(A, j, 0) = at(A, 0, j);
		for (i = 0; i < j - 1; i++) {
			at(A, i, j) = 0;
		}
	}
	return 0;
}

#define n 6

int main()
{
	int i;
#if 0
	float A_data[n * n] = { 0 };
	FILE* A_file = fopen("../A.bin", "rb");
	fread(A_data, sizeof(float), n * n, A_file); 
	fclose(A_file);
#else
	float A_data[n * n] = { 
		 92,   85,   -4, -60,  -40,  55,
		 85, -147,   44, -84, -111,  97,
		 -4,   44,  152,  86, -125,  23,
		-60,  -84,   86, 334,   -4, -99,
		-40, -111, -125,  -4,  173, -93,
		 55,   97,   23, -99,  -93,  93 
	};
#endif
	float A_copy[n * n] = { 0 };
	float P_data[n * n] = { 0 };
	float T_data[n * n] = { 0 };
	float B_data[n * n] = { 0 };
	Matrixf A = { n, n, A_data };
	Matrixf P = { n, n, P_data };
	Matrixf T = { n, n, T_data };
	Matrixf B = { n, n, B_data };

	matrixf_init(&A, n, n, A_data, 1);
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	for (i = 0; i < n * n; i++) A_copy[i] = A_data[i];
	matrixf_decomp_ltl(&A);
	P.data[0] = 1;
	for (i = 0; i < n; i++) {
		if (i > 0) {
			at(&P, i, (int)at(&A, i, 0)) = 1;
			at(&A, i, 0) = 0;
		}
		at(&T, i, i) = at(&A, i, i);
		at(&A, i, i) = 1;
		if (i < n - 1) {
			at(&T, i + 1, i) = at(&A, i, i + 1);
			at(&T, i, i + 1) = at(&A, i, i + 1);
			at(&A, i, i + 1) = 0;
		}
	}
	printf("L = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	printf("T = \n"); matrixf_print(&T, "%9.4f "); printf("\n");
	printf("P = \n"); matrixf_print(&P, "%3.0f "); printf("\n");

	matrixf_multiply(&P, &A, &B, 1, 0, 1, 0);
	matrixf_multiply(&B, &T, &P, 1, 0, 0, 0);
	matrixf_multiply(&P, &B, &A, 1, 0, 0, 1);
	printf("P'*L*T*L'*P = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	for (i = 0; i < n * n; i++) A_copy[i] -= A.data[i];
	printf("||A - P'*L*T*L'*P||_F = %g\n\n", normf(A_copy, n * n, 1));

	return 0;
}