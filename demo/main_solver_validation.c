/*
clear; close all; clc;

m = 200;
n = 180;
d = 40;

p = "./";

A = single(randn(m, n));
b = single(randn(size(A, 1), d));

tic
x = A\b;
toc

fileID = fopen(p + "A.bin",'w');
fwrite(fileID, A, 'single');
fclose(fileID);
fileID = fopen(p + "b.bin",'w');
fwrite(fileID, b, 'single');
fclose(fileID);

disp('Run the program and press "enter" to continue...'); pause()

fileID = fopen(p + "x.bin",'r');
xc = fread(fileID, size(x), 'single');
fclose(fileID);

figure; plot(x(:)-xc(:))
*/

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
#include "stdio.h"
#include "stdlib.h"
#include "detectum.h"

#define M 200
#define N 180
#define P 40

int main()
{
	FILE* A_file = fopen("../A.bin", "rb");
	FILE* b_file = fopen("../b.bin", "rb");
	FILE* x_file = fopen("../x.bin", "wb");
	Matrixf A = matrixf(M, N), B = matrixf(M, P);
	Matrixf X = { { N, P }, B.data };
#ifdef TICKTOCK
	struct timespec t0;
#endif
	if (!A.data || !B.data) return -1;
	fread(A.data, sizeof(float), (size_t)(A.size[0] * A.size[1]), A_file);
	fread(B.data, sizeof(float), (size_t)(B.size[0] * B.size[1]), b_file);
	fclose(A_file);
	fclose(b_file);
#ifdef TICKTOCK
	tick(&t0);
#endif
	if (matrixf_solve_qr(&A, &B, &X)) return -1;
#ifdef TICKTOCK
	printf("Elapsed time: %f s\n", tock(&t0));
#endif
	fwrite(X.data, sizeof(float), (size_t)(X.size[0] * X.size[1]), x_file);
	fclose(x_file);
	free(A.data);
	free(B.data);

	return 0;
}