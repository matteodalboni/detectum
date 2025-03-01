#include "detego.h"
#include "detego_utils.h"

int main()
{
	float A_data[3 * 3] = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
		0.7f, 0.8f, 0.9f
	};
	float work[2 * 3 * 3] = { 0 };
	Matrixf A;
	unsigned int p = 15;

	matrixf_init(&A, 3, 3, A_data, 1);
	DISP("%9.4f ", A);
	matrixf_pow(&A, p, work);
	printf("A^%d = \n", p); PRINT("%9.4f ", &A);

	return 0;
}