#ifndef DETEGO_UTILS_H
#define DETEGO_UTILS_H

#include <stdio.h>

static inline void print_data(char* format, float* data, int rows, int cols)
{
	int i, j;

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			printf(format, data[i + j * rows]);
		} printf("\n");
	} printf("\n");
}

#define PRINT(format, A) \
print_data(format, (A)->data, (A)->size[0], (A)->size[1])

#define DISP(format, A) \
do{ printf(#A" = \n"); PRINT(format, &A); } while (0)

#endif