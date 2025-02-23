#include "verto.h"
#include "verto_utils.h"
#include "verto_alloc.h"

#define M 2002
#define N 2001
#define P 2000

#define PRINT_MATRIX 0
#define NO_TRANSPOSE 1

int main()
{
	int i, transA = 1, transB = 1;
	float alpha = 2, beta = 3;
	Matrixf A = MATRIXF(M, P), B = MATRIXF(P, N), C = MATRIXF(M, N);
#ifdef TIME_TICK_H
	struct timespec t0;
#endif

	for (i = 0; i < M * P; i++) A.data[i] = (float)i;
	for (i = 0; i < P * N; i++) B.data[i] = (float)i;
	for (i = 0; i < M * N; i++) C.data[i] = (float)i;

	if (transA) matrixf_transpose(&A);
	if (transB) matrixf_transpose(&B);

#if PRINT_MATRIX
	DISP("%7.0f ", A);
	DISP("%7.0f ", B);
	DISP("%7.0f ", C);
#endif
#ifdef TIME_TICK_H
	tick(&t0);
#endif
#if NO_TRANSPOSE
	if (!matrixf_multiply(&A, &B, &C, alpha, beta, transA, transB)) {
#ifdef TIME_TICK_H
		printf("Elapsed time: %f s\n\n", tock(&t0));
#endif
	}
#else
	if (transA) matrixf_transpose(&A);
	if (transB) matrixf_transpose(&B);
	if (!matrixf_multiply(&A, &B, &C, alpha, beta, 0, 0)) {
		if (transA) matrixf_transpose(&A);
		if (transB) matrixf_transpose(&B);
#ifdef TIME_TICK_H
		printf("Elapsed time: %f s\n\n", tock(&t0));
#endif
	}
#endif
	else {
		printf("ERROR: size mismatch\n\n");
	}
#if PRINT_MATRIX
	printf("%g * A%s * B%s + %g * C = \n", alpha,
		transA ? "'" : "", transB ? "'" : "", beta);
	PRINT("%7.0f ", &C);
#endif

	free(A.data);
	free(B.data);
	free(C.data);

	return 0;
}