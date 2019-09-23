#ifndef GPUALGO_H_3PGDZ4GM
#define GPUALGO_H_3PGDZ4GM

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#include "gpuUtils.h"
#include "gpuTimer.h"

__global__ void gpuDeconvolute(double *a, double *b, double *c, double f, uint32_t n);
__global__ void gpuOffsetDifferentiate(double *a, double *b, uint32_t gap, uint32_t n);
__global__ void gpuMovingAverage(double *a, double *b, uint32_t window, uint32_t n);

#endif /* end of include guard: GPUALGO_H_3PGDZ4GM */
