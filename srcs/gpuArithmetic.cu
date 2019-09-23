#include "gpuArithmetic.h"

__global__ void gpuScale(double *a, double *b, double f, uint32_t n) {
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if (gId < n)
    b[gId] = f * a[gId];
}

__global__ void gpuAdd(double *a, double *b, double *c, uint32_t n) {
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if (gId < n)
    c[gId] = a[gId] + b[gId];
}

__global__ void gpuSubtract(double *a, double *b, double *c, uint32_t n) {
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if (gId < n)
    c[gId] = a[gId] - b[gId];
}

__global__ void gpuDeconvolute(double *a, double *b, double *c, double f, uint32_t n){
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if (gId < n)
    c[gId] = a[gId] - f * b[gId];
}
