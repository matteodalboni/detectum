#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

static void print_complex_eigenvec_matrix(Matrixf* T, Matrixf* V,
	const char* format_real, const char* format_cplx)
{
	size_t i, j;
	const size_t n = T->rows;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (j == n - 1 || at(T, j + 1, j) == 0) {
				printf(format_real, at(V, i, j));
			}
			else {
				printf(format_cplx, at(V, i, j), +at(V, i, j + 1));
				printf(format_cplx, at(V, i, j), -at(V, i, j + 1));
				j++;
			}
		}
		printf("\n");
	}
}

static void print_complex_eigenval_matrix(Matrixf* T_blkdiag,
	const char* format_real, const char* format_cplx)
{
	size_t i, j;
	int cplx = 0;
	const size_t n = T_blkdiag->rows;
	float re, im;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (!cplx && (j == n - 1 || at(T_blkdiag, j + 1, j) == 0)) {
				re = i == j ? at(T_blkdiag, i, j) : 0;
				im = 0;
				printf(format_real, re);
			}
			else {
				if (!cplx) {
					re = i == j ? at(T_blkdiag, i, j) : 0;
					im = i == j ? at(T_blkdiag, i, j + 1) : 0;
					cplx = 1;
				}
				else {
					re = i == j ? at(T_blkdiag, i, j) : 0;
					im = i == j ? at(T_blkdiag, i, j - 1) : 0;
					cplx = 0;
				}
				printf(format_cplx, re, im);
			}
		}
		printf("\n");
	}
}

static void get_eigenval_matrix(Matrixf* T)
{
	size_t i, j;
	const size_t n = T->rows;

	for (j = 0; j < n; j++) {
		if (j == n - 1 || at(T, j + 1, j) == 0) {
			for (i = 0; i < j; i++) {
				at(T, i, j) = 0;
			}
		}
		else {
			for (i = 0; i < j; i++) {
				at(T, i, j) = at(T, i, j + 1) = 0;
			}
			at(T, j + 1, j) = -sqrtf(-at(T, j + 1, j) * at(T, j, j + 1));
			at(T, j, j + 1) = -at(T, j + 1, j);
			j++;
		}
	}
}

int main()
{
	const int test = 1;
	const int pseudo = 0;
	const char format_r[] = " %10.4g";
	const char format_c[] = " %10.4f%+.4fi";
	int i = 0, j = 0, n, exitflag;
	float* work, a = 0, b = 0;
	Matrixf A, U, V, W;
	FILE* file = 0; // freopen("../output.m", "w", stdout);

	switch (test) {
	case 0:
		n = 10;
		A = matrixf(n, n);
		for (j = 0; j < n; j++) {
			for (i = 0; i < n; i++) {
				at(&A, i, j) = i == j ? -3.0f : 0.0f;
				if (j > 0 && i == j - 1) {
					at(&A, i, j) = 100.0f;
				}
			}
		}
		//matrixf_transpose(&A);
		break;
	case 1:
		n = 8;
		A = matrixf(n, n);
		srand(1992);
		for (i = 0; i < n * n; i++) {
			A.data[i] = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
		}
		break;
	case 2:
		n = 3;
		A = matrixf(n, n);
		at(&A, 0, 0) = 1; at(&A, 0, 1) = 1; at(&A, 0, 2) = 1;
		at(&A, 1, 0) = 0; at(&A, 1, 1) = 1; at(&A, 1, 2) = 0;
		at(&A, 2, 0) = 0; at(&A, 2, 1) = 0; at(&A, 2, 2) = 1;
		break;
	case 3:
		n = 8;
		A = matrixf(n, n);
		a = 3;
		b = -10;
		at(&A, 0, 0) = +a; at(&A, 0, 1) = -b; at(&A, 0, 2) = +1; at(&A, 0, 3) = +0;
		at(&A, 1, 0) = +b; at(&A, 1, 1) = +a; at(&A, 1, 2) = +0; at(&A, 1, 3) = +1;
		at(&A, 2, 0) = +0; at(&A, 2, 1) = +0; at(&A, 2, 2) = +a; at(&A, 2, 3) = -b;
		at(&A, 3, 0) = +0; at(&A, 3, 1) = +0; at(&A, 3, 2) = +b; at(&A, 3, 3) = +a;
		at(&A, 4, 4) = +a; at(&A, 4, 5) = -b; at(&A, 4, 6) = +1; at(&A, 4, 7) = +0;
		at(&A, 5, 4) = +b; at(&A, 5, 5) = +a; at(&A, 5, 6) = +0; at(&A, 5, 7) = +1;
		at(&A, 6, 4) = +0; at(&A, 6, 5) = +0; at(&A, 6, 6) = +a; at(&A, 6, 7) = -b;
		at(&A, 7, 4) = +0; at(&A, 7, 5) = +0; at(&A, 7, 6) = +b; at(&A, 7, 7) = +a;
		break;
	}
	printf("\nA = [\n"); matrixf_print(&A, format_r); printf("];\n");
	U = matrixf(n, n);
	V = matrixf(n, n);
	W = matrixf(n, n);
	matrixf_decomp_schur(&A, &U);
	printf("\nT = [\n"); matrixf_print(&A, format_r); printf("];\n");
	printf("\nU = [\n"); matrixf_print(&U, format_r); printf("];\n");
	work = malloc(sizeof(float) * ((size_t)4 * n * n + (size_t)2 * n));
	exitflag = matrixf_get_eigenvectors(&A, &U, &V, &W, pseudo, work);
	if (pseudo) {
		printf("\nV = [\n"); matrixf_print(&V, format_r); printf("];\n");
		printf("\nW = [\n"); matrixf_print(&W, format_r); printf("];\n");
	}
	else {
		printf("\nV = [\n"); print_complex_eigenvec_matrix(&A, &V, format_r, format_c); printf("];\n");
		printf("\nW = [\n"); print_complex_eigenvec_matrix(&A, &W, format_r, format_c); printf("];\n");
	}
	get_eigenval_matrix(&A);
	if (pseudo) {
		printf("\nD = [\n"); matrixf_print(&A, format_r); printf("];\n");
	}
	else {
		printf("\nD = [\n"); print_complex_eigenval_matrix(&A, format_r, format_c); printf("];\n");
	}
	if (file) {
		printf("\nclc \nnorm(A - U*T*U') \nnorm(A*V - V*D) \nnorm(W'*A - D*W') \n");
	}
	free(A.data);
	free(U.data);
	free(V.data);
	free(W.data);
	free(work);
	return exitflag;
}