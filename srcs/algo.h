#ifndef ALGO_HH_8GE72YHP
#define ALGO_HH_8GE72YHP

#include <stdint.h>
#include "vector.h"

Vector * Deconvolute(Vector * wf, double f);
Vector * OffsetDifferentiate(Vector * wf, uint32_t M);
Vector * MovingAverage(Vector * wf, uint32_t L);
Vector * MWD(Vector * wf,  double f, uint32_t M, uint32_t L);

#endif /* end of include guard: ALGO_HH_8GE72YHP */
