#ifndef VECTOR_H_RNBKCSK5
#define VECTOR_H_RNBKCSK5

#include<stdint.h>

#define VECTOR_INITIAL_CAPACITY 3000
#define VECTOR_GROWING_FACTOR 1.6

typedef struct Vector {
  uint32_t size;
  uint32_t capacity;
  double * data;
} Vector;

Vector * VectorInit();
void VectorAppend(Vector *vector, double value);
double VectorGet(Vector *vector, uint32_t index);
void VectorSet(Vector *vector, uint32_t index, double value);
void VectorExpandIfFull(Vector *vector);
void VectorExpand(Vector *vector);
void VectorFree(Vector *vector);
void VectorCopy(Vector *dest, Vector *src, uint32_t start, uint32_t stop);

#endif /* end of include guard: VECTOR_H_RNBKCSK5 */
