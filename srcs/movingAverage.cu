#include "movingAverage.h"

__global__ void gpuMovingAverage(double *a, double *b, uint32_t window, uint32_t n) {
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if ((gId < n) && (gId >= window)) {
    double sum = 0.;
    for (int i = gId; i < gId + window; ++i) {
      sum += a[i];
    }
    b[gId] = sum / window;
  }
}
