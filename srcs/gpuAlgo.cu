#include "gpuAlgo.h"

__global__ void gpuDeconvolute(double *a, double *b, double *c, double f, uint32_t n){
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if (gId < n)
    c[gId] = a[gId] + f * b[gId];
}

__global__ void gpuOffsetDifferentiate(double *a, double *b, uint32_t gap, uint32_t n){
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if (gId < gap) {
    b[gId] = 0.;
  }
  else if (gId < n) {
    b[gId] = a[gId] - a[gId - gap];
  }
}

__global__ void gpuMovingAverage(double *a, double *b, uint32_t window, uint32_t n){
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  double sum = 0.;
  if (gId < window) {
    b[gId] = a[gId];
  }
  else if (gId < n){
    sum = 0.;
    for (uint32_t i = 0; i < window; ++i) {
      sum += a[gId - i];
    }
    b[gId] = sum / window;
  }
}
