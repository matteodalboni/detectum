#include <stdio.h>
#include "detectum.h"

static void print_complex(float* Real, float* Imag, int rows, int cols, char* format) {

	int i, j;

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			printf(format, Real[i * cols + j], Imag[i * cols + j]);
		}
		printf("\n");
	}
}

#define m 4
#define n 3

int main() 
{
	float A_real_data[] = {
		-5, -8,  2, 
		 4,  0, -6, 
		 3, 10,  5, 
		-7, -3, -5
	};
	float A_imag_data[] = {
		 0,  1,  7,
		 4, -8, -5,
		 8, -7,  7,
		10, -5, -5
	};
	float b_real_data[] = {
		  9,
		 -3,
		 -6,
		 -5
	};
	float b_imag_data[] = {
		 2,
		-1,
		-3,
		 7
	};
	float A_data[4 * m * n] = { 0 };
	float b_data[2 * m] = { 0 };
	Matrixf A_real, A_imag, b_real, b_imag;
	Matrixf A = { { 2 * m, 2 * n }, A_data };
	Matrixf b = { { 2 * m, 1 }, b_data };
	Matrixf x = { { 2 * n, 1 }, b_data };
	int i, j;

	printf("\nA = \n"); print_complex(A_real_data, A_imag_data, m, n, "%5.0f%+.0fi\t"); 
	printf("\nb = \n"); print_complex(b_real_data, b_imag_data, m, 1, "%5.0f%+.0fi\t"); 

	matrixf_init(&A_real, m, n, A_real_data, 1);
	matrixf_init(&A_imag, m, n, A_imag_data, 1);
	matrixf_init(&b_real, m, 1, b_real_data, 1);
	matrixf_init(&b_imag, m, 1, b_imag_data, 1);

	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			at(&A, i + 0, j + 0) = +at(&A_real, i, j);
			at(&A, i + m, j + 0) = +at(&A_imag, i, j);
			at(&A, i + 0, j + n) = -at(&A_imag, i, j);
			at(&A, i + m, j + n) = +at(&A_real, i, j);
		}
	}
	for (i = 0; i < m; i++) {
		at(&b, i + 0, 0) = +at(&b_real, i, 0);
		at(&b, i + m, 0) = +at(&b_imag, i, 0);
	}

	//printf("\nA = \n"); matrixf_print(&A, "%5.0f ");
	//printf("\nb = \n"); matrixf_print(&b, "%5.0f ");
	matrixf_solve_qr(&A, &b, &x);
	//printf("\nx = \n"); matrixf_print(&x, "%9.4f ");
	printf("\nThe least-squares solution to the complex system A*x=b is \n");
	printf("\nx = \n"); print_complex(x.data, x.data + n, n, 1, "%9.4f%+.4fi\t"); 

	return 0;
}