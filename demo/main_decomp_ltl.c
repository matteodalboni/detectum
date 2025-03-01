#include "detego.h"
#include "detego_utils.h"

#define n 6

int main()
{
	int i;
	float A_data[n * n] = { 
		 92,   85,   -4, -60,  -40,  55,
		 85, -147,   44, -84, -111,  97,
		 -4,   44,  152,  86, -125,  23,
		-60,  -84,   86, 334,   -4, -99,
		-40, -111, -125,  -4,  173, -93,
		 55,   97,   23, -99,  -93,  93 
	};
	float A_copy[n * n] = { 0 };
	float P_data[n * n] = { 0 };
	float T_data[n * n] = { 0 };
	float B_data[n * n] = { 0 };
	Matrixf A = { {n, n}, A_data };
	Matrixf P = { {n, n}, P_data };
	Matrixf T = { {n, n}, T_data };
	Matrixf B = { {n, n}, B_data };

	matrixf_init(&A, n, n, A_data, 1);
	DISP("%9.4f ", A);
	for (i = 0; i < n * n; i++) A_copy[i] = A_data[i];
	matrixf_decomp_ltl(&A);
	P.data[0] = 1;
	for (i = 1; i < n; i++) {
		_(&P, i, (int)_(&A, i, 0)) = 1;
		_(&A, i, 0) = 0;
	}
	for (i = 0; i < n; i++) {
		_(&T, i, i) = _(&A, i, i);
		_(&A, i, i) = 1;
		if (i < n - 1) {
			_(&T, i + 1, i) = _(&A, i, i + 1);
			_(&T, i, i + 1) = _(&A, i, i + 1);
			_(&A, i, i + 1) = 0;
		}
	}
	printf("L = \n"); PRINT("%9.4f ", &A);
	DISP("%9.4f ", T);
	DISP("%3.0f ", P);

	matrixf_multiply(&P, &A, &B, 1, 0, 1, 0);
	matrixf_multiply(&B, &T, &P, 1, 0, 0, 0);
	matrixf_multiply(&P, &B, &A, 1, 0, 0, 1);
	printf("P'*L*T*L'*P = \n"); PRINT("%9.4f ", &A);
	for (i = 0; i < n * n; i++) A_copy[i] -= A.data[i];
	printf("||A - P'*L*T*L'*P||_F = %g\n\n", get_norm2(A_copy, n * n, 1));

	return 0;
}