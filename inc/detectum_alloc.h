#ifndef DETECTUM_ALLOC_H
#define DETECTUM_ALLOC_H

#include "detectum.h"
#include <stdlib.h>

// This function initializes a rows-by-cols matrix, allocating its data 
// memory on the heap. On allocation failure, the data pointer is null.
static inline Matrixf matrixf(size_t rows, size_t cols)
{
  Matrixf A = { rows, cols, calloc(rows * cols, sizeof(float)) };

  return A;
}

#endif
