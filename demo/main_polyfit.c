#include <stdio.h>
#include "detego.h"

#define PI 3.141592653589793f
#define PTS 10
#define DEG 7
#define WORK_LEN (PTS * (DEG + 1))

// This function performs the polynomial fitting of the points with
// coordinates x and y. pts is the number of points, and deg is the
// degree of the polynomial. The vector y is overwritten by the deg+1
// polynomial coefficients, which are in descending powers. The array
// work is the  additional workspace memory: its minimum length is 
// pts*(deg+1).
int polyfit(float* x, float* y, int pts, int deg, float* work)
{
	int i, j;
	const int m = pts;
	const int n = deg + 1;
	Matrixf A = { { m, n }, work };
	Matrixf b = { { m, 1 }, y };

	for (j = 0; j < n; j++)
		for (i = 0; i < m; i++)
			at(&A, i, j) = powf(x[i], (float)(n - 1 - j));

	return matrixf_solve_lsq(&A, &b);
}

int main()
{
	int i;
	float x[PTS] = { 0 };
	float y[PTS] = { 0 };
	float work[WORK_LEN] = { 0 };

	for (i = 0; i < PTS; i++)
	{
		x[i] = i * 4.0f * PI / (float)(PTS - 1);
		y[i] = sinf(x[i]);
	}

	if (polyfit(x, y, PTS, DEG, work) < 0) return -1;

	printf("p = @(x) "); 
	for (i = 0; i <= DEG; i++) 
		printf("%+3.10f*x.^%d", y[i], DEG - i);

	return 0;
}