#ifndef GPUARITHMETIC_H_1OSHUQJR
#define GPUARITHMETIC_H_1OSHUQJR

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#include "gpuUtils.h"
#include "gpuTimer.h"

__global__ void gpuAdd(double *a, double *b, double *c, uint32_t n);
__global__ void gpuSubtract(double *a, double *b, double *c, uint32_t n);
__global__ void gpuScale(double *a, double *b, double f, uint32_t n);
__global__ void gpuDeconvolute(double *a, double *b, double *c, double f, uint32_t n);

#endif /* end of include guard: GPUARITHMETIC_H_1OSHUQJR */

