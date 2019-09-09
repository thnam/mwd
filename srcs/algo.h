#ifndef ALGO_HH_8GE72YHP
#define ALGO_HH_8GE72YHP

#define VECTOR_INITIAL_CAPACITY 3000
#define VECTOR_GROWING_FACTOR 1.6

#include <stdint.h>
typedef struct Vector {
  uint32_t size;
  uint32_t capacity;
  double * data;
} Vector;

Vector * Deconvolute(Vector * wf, double f);
Vector * OffsetDifferentiate(Vector * wf, uint32_t M);
Vector * MovingAverage(Vector * wf, uint32_t L);
Vector * MWD(Vector * wf,  double f, uint32_t M, uint32_t L);

Vector * VectorInit();
void VectorAppend(Vector *vector, double value);
double VectorGet(Vector *vector, uint32_t index);
void VectorSet(Vector *vector, uint32_t index, double value);
void VectorExpandIfFull(Vector *vector);
void VectorExpand(Vector *vector);
void VectorFree(Vector *vector);
void VectorCopy(Vector *dest, Vector *src, uint32_t start, uint32_t stop);

#endif /* end of include guard: ALGO_HH_8GE72YHP */
