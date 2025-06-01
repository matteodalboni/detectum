#define TICKTOCK

#ifdef TICKTOCK
#include <time.h>
static inline int tick(struct timespec* t0) {
#ifdef _WIN32
	if (timespec_get(t0, TIME_UTC) == TIME_UTC) return 1;
#else
	if (clock_gettime(CLOCK_REALTIME, t0) == 0) return 1;
#endif
	return 0;
}
static inline double tock(struct timespec* t0) {
	struct timespec tf = { 0 }; tick(&tf);
	return ((tf.tv_sec - t0->tv_sec) + ((tf.tv_nsec - t0->tv_nsec) * 1e-9));
}
#endif
#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

#define M 2002
#define N 2001
#define P 2000

#define PRINT_MATRIX 0
#define NO_TRANSPOSE 1

int main()
{
	int i, transA = 1, transB = 1;
	float alpha = 2, beta = 3;
	Matrixf A = matrixf(M, P), B = matrixf(P, N), C = matrixf(M, N);
#ifdef TICKTOCK
	struct timespec t0;
#endif

	for (i = 0; i < M * P; i++) A.data[i] = (float)i;
	for (i = 0; i < P * N; i++) B.data[i] = (float)i;
	for (i = 0; i < M * N; i++) C.data[i] = (float)i;

	if (transA) matrixf_transpose(&A);
	if (transB) matrixf_transpose(&B);

#if PRINT_MATRIX
	printf("A = \n"); matrixf_print(&A, "%7.0f "); printf("\n");
	printf("B = \n"); matrixf_print(&B, "%7.0f "); printf("\n");
	printf("C = \n"); matrixf_print(&C, "%7.0f "); printf("\n");
#endif
#ifdef TICKTOCK
	tick(&t0);
#endif
#if NO_TRANSPOSE
	if (!matrixf_multiply(&A, &B, &C, alpha, beta, transA, transB)) {
#ifdef TICKTOCK
		printf("Elapsed time: %f s\n\n", tock(&t0));
#endif
	}
#else
	if (transA) matrixf_transpose(&A);
	if (transB) matrixf_transpose(&B);
	if (!matrixf_multiply(&A, &B, &C, alpha, beta, 0, 0)) {
		if (transA) matrixf_transpose(&A);
		if (transB) matrixf_transpose(&B);
#ifdef TICKTOCK
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
	matrixf_print(&C, "%7.0f "); printf("\n");
#endif

	free(A.data);
	free(B.data);
	free(C.data);

	return 0;
}