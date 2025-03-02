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

#define _CRT_SECURE_NO_WARNINGS

#define DETEGO_USE_PRINT
#define DETEGO_USE_ALLOC
#include "detego.h"
//#include "time_tick.h"

#define M 200
#define N 180
#define P 40

int main()
{
	FILE* A_file = fopen("../A.bin", "rb");
	FILE* b_file = fopen("../b.bin", "rb");
	FILE* x_file = fopen("../x.bin", "wb");
	Matrixf A = matrixf(M, N), B = matrixf(M, P);
#ifdef TIME_TICK_H
	struct timespec t0;
#endif
	if (!A.data || !B.data) return -1;
	fread(A.data, sizeof(float), (size_t)(A.size[0] * A.size[1]), A_file);
	fread(B.data, sizeof(float), (size_t)(B.size[0] * B.size[1]), b_file);
	fclose(A_file);
	fclose(b_file);
#ifdef TIME_TICK_H
	tick(&t0);
#endif
	if (matrixf_solve_lsq(&A, &B)) return -1;
#ifdef TIME_TICK_H
	printf("Elapsed time: %f s\n", tock(&t0));
#endif
	fwrite(B.data, sizeof(float), (size_t)(B.size[0] * B.size[1]), x_file);
	fclose(x_file);
	free(A.data);
	free(B.data);

	return 0;
}