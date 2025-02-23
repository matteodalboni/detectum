/*This example is taken from Matlab documentation:
https://it.mathworks.com/help/matlab/math/systems-of-linear-equations.html
*/

#include <stdio.h>
#include "verto.h"

#define M 6
#define N 2

int main()
{
	int i;
	float t[M] = { 0.0f, 0.3f, 0.8f, 1.1f, 1.6f, 2.3f };
	float y[M] = { 0.82f, 0.72f, 0.63f, 0.60f, 0.55f, 0.50f };
	float data[M * N] = { 0 };
	Matrixf E = { 0 }, Y = { 0 };

	matrixf_init(&E, M, N, data, 0);
	matrixf_init(&Y, M, 1, y, 0);

	for (i = 0; i < M; i++)
	{
		_(&E, i, 0) = 1;
		_(&E, i, 1) = expf(-t[i]);
	}
		
	matrixf_solve_lsq(&E, &Y);

	printf("The model is y(t) = %f%+f*exp(-t)", y[0], y[1]);

	return 0;
}