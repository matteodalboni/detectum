#ifndef VERTO_ALLOC_H
#define VERTO_ALLOC_H

#include <stdlib.h>

#define MATRIXF(rows, cols) \
{ rows, cols, calloc(sizeof(float), (rows) * (cols)) }

#endif