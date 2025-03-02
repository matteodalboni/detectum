#ifndef DETEGO_ALLOC_H
#define DETEGO_ALLOC_H

#include <stdlib.h>

#define matrixf(rows, cols) \
{ rows, cols, calloc(sizeof(float), (rows) * (cols)) }

#endif