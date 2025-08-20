#include <stdio.h>
#include "detectum.h"

#define m 8
#define n 6

int main()
{
	float A_data[] = {
        64,   2,   3,  61,  60,   6,
         9,  55,  54,  12,  13,  51,
        17,  47,  46,  20,  21,  43,
        40,  26,  27,  37,  36,  30,
        32,  34,  35,  29,  28,  38,
        41,  23,  22,  44,  45,  19,
        49,  15,  14,  52,  53,  11,
         8,  58,  59,   5,   4,  62
	};
    float b_data[] = { 
        260, 260, 260, 260, 260, 260, 260, 260 
    };
    float work[n * (n + 1)];
    Matrixf A, b, x = { n, 1, work };
    
    matrixf_init(&A, m, n, A_data, 1);
    matrixf_init(&b, m, 1, b_data, 0);
    printf("A = \n"); matrixf_print(&A, "%9.0f"); printf("\n");
    matrixf_pseudoinv(&A, -1, work);
    printf("pinv(A) = \n"); matrixf_print(&A, "%9.4f"); printf("\n");
    matrixf_multiply(&A, &b, &x, 1, 0, 0, 0);
    printf("x = \n"); matrixf_print(&x, "%9.4f"); printf("\n");

	return 0;
}