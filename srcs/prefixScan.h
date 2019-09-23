#ifndef SCAN_H__
#define SCAN_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#include "utils.h"
#include "timer.h"

void sum_scan_blelloch(double* const d_out, const double* const d_in,
	const size_t numElems);

#endif
