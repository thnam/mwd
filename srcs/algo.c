#include "algo.h"
#include<stdlib.h>
#include<stdio.h>
#include<string.h>

Vector * Deconvolute(Vector * wf, double f){
  if (wf->size <= 2) {
    return wf;
  }

  // the output waveform has the same length as that of the input
  Vector * A = VectorInit();

  VectorAppend(A, wf->data[0]);

  for (uint32_t i = 1; i < wf->size; ++i) {
    VectorAppend(A, wf->data[i] - f*wf->data[i-1] + A->data[i-1]);
  }

  return A;
}

Vector * OffsetDifferentiate(Vector * wf, uint32_t M){
  if (wf->size <= M) {
    return wf;
  }

  // size of original waveform less M data
  Vector * D = VectorInit();
  /* Vector * D = (Vector *) malloc(2*sizeof(uint32_t) + (wf->size - M) * sizeof(double)); */
  for (uint32_t i = M; i < wf->size; ++i) {
    VectorAppend(D, wf->data[i] - wf->data[i - M]);
  }
  return D;
}

Vector * MovingAverage(Vector * wf, uint32_t L){
  if (wf->size <= L) {
    return wf;
  }

  double sum = 0.;
  /* Vector * MA = (Vector *) malloc(2*sizeof(uint32_t) + (wf->size - L) * sizeof(double)); */
  Vector * MA = VectorInit();
  for (uint32_t i = 0; i < L; ++i) {
    sum += wf->data[i];
  }
  VectorAppend(MA, sum / L);

  for (uint32_t i = L; i < wf->size; ++i) {
    sum += wf->data[i] - wf->data[i - L];
    VectorAppend(MA, sum / L);
  }
  return MA;
}

Vector * MWD(Vector * wf,  double f, uint32_t M, uint32_t L){
  Vector * deconv = Deconvolute(wf, f);
  Vector * odiff = OffsetDifferentiate(deconv, M);
  Vector * mavg = MovingAverage(odiff, L);
  VectorFree(odiff);
  VectorFree(deconv);
  return mavg;
}
