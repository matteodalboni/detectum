#include <stdio.h>
#include "detectum.h"

#define deg 4 // degree of the polynomial

int main()
{
	int i;
	float re, im;
	float coeffs[deg + 1] = { 1, 0, 0, 0, -1 };

	// Compute roots
	Matrixf(A, deg, deg);
	at(&A, 0, 0) = -coeffs[1] / coeffs[0];
	for (i = 0; i < deg - 1; i++) {
		at(&A, 0, i + 1) = -coeffs[i + 2] / coeffs[0];
		at(&A, i + 1, i) = 1.0f;
	}
	matrixf_decomp_schur(&A, 0);
	// Print roots
	printf("Roots of");
	for (i = 0; i <= deg; i++) {
		printf(" %c ", coeffs[i] < 0 ? '-' : '+');
		printf("%g*x^%d", fabsf(coeffs[i]), deg - i);
	}
	printf("\n");
	i = 0;
	while (i < deg) {
		re = at(&A, i, i);
		if (i == deg - 1 || at(&A, i + 1, i) == 0) {
			printf("%d) %+.4f\n", i + 1, re);
			i += 1;
		}
		else {
			im = sqrtf(-at(&A, i + 1, i) * at(&A, i, i + 1));
			printf("%d) %+.4f%+.4fi\n", i + 1, re, -im);
			printf("%d) %+.4f%+.4fi\n", i + 2, re, +im);
			i += 2;
		}
	}
	return 0;
}