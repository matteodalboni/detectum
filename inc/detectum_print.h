#ifndef DETECTUM_PRINT_H
#define DETECTUM_PRINT_H

#include "detectum.h"
#include <stdio.h>

// This function prints the matrix A on the standard output according to
// the specified format.
static inline void matrixf_print(Matrixf* A, const char* format)
{
  size_t i, j;

  for (i = 0; i < A->rows; i++) {
    for (j = 0; j < A->cols; j++) {
      printf(format, at(A, i, j));
    }
    printf("\n");
  }
}

#endif
