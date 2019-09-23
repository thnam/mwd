#ifndef MOVINGAVERAGE_H_SVZYPTJO
#define MOVINGAVERAGE_H_SVZYPTJO

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#include "gpuUtils.h"
#include "gpuTimer.h"

__global__ void gpuMovingAverage(double *a, double *b, uint32_t window, uint32_t n);

#endif /* end of include guard: MOVINGAVERAGE_H_SVZYPTJO */
