#define DETEGO_USE_PRINT
#include "detego.h"

int main()
{
	float A_data[8 * 6] = {
        64,   2,   3,  61,  60,   6,
         9,  55,  54,  12,  13,  51,
        17,  47,  46,  20,  21,  43,
        40,  26,  27,  37,  36,  30,
        32,  34,  35,  29,  28,  38,
        41,  23,  22,  44,  45,  19,
        49,  15,  14,  52,  53,  11,
         8,  58,  59,   5,   4,  62
	};
    float b_data[8] = { 
        260, 260, 260, 260, 260, 260, 260, 260 
    };
    float work[6 * 7];
    Matrixf A, b, x = { { 6, 1 }, work };
    
    matrixf_init(&A, 8, 6, A_data, 1);
    matrixf_init(&b, 8, 1, b_data, 0);
    printf("A = \n"); matrixf_print(&A, "%9.0f");
    matrixf_pseudoinv(&A, -1, work);
    printf("A = \n"); matrixf_print(&A, "%9.4f");
    matrixf_multiply(&A, &b, &x, 1, 0, 0, 0);
    printf("x = \n"); matrixf_print(&x, "%9.4f");

	return 0;
}