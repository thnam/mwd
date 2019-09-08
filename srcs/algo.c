#include "algo.h"
#include<stdlib.h>
#include<stdio.h>

Vector * Deconvolute(Vector * wf, double f){
  if (wf->size <= 2) {
    return wf;
  }

  // the output waveform has the same length as that of the input
  Vector * A = (Vector*)malloc(sizeof(uint32_t) + sizeof(double) * wf->size);

  A->samples[0] = wf->samples[0];
  A->size = 1;

  for (uint32_t i = 1; i < wf->size; ++i) {
    A->samples[i] = wf->samples[i] - f*wf->samples[i-1] + A->samples[i-1];
    A->size++;
  }

  return A;
}

Vector * OffsetDifferentiate(Vector * wf, uint32_t M){
  if (wf->size <= M) {
    return wf;
  }

  // size of original waveform less M samples
  Vector * D = (Vector *) malloc(sizeof(uint32_t) + (wf->size - M) * sizeof(double));
  D->size = 0;
  for (uint32_t i = M; i < wf->size; ++i) {
    D->samples[i - M] = wf->samples[i] - wf->samples[i - M];
    D->size++;
  }
  return D;
}

Vector * MovingAverage(Vector * wf, uint32_t L){
  if (wf->size <= L) {
    return wf;
  }

  double sum = 0.;
  Vector * MA = (Vector *) malloc(sizeof(uint32_t) + (wf->size - L) * sizeof(double));
  for (uint32_t i = 0; i < L; ++i) {
    sum += wf->samples[i];
  }
  MA->samples[0] = sum / L;
  MA->size = 1;

  for (uint32_t i = L; i < wf->size; ++i) {
    sum += wf->samples[i] - wf->samples[i - L];
    MA->samples[i - L + 1] = sum / L;
    MA->size++;
  }
  return MA;
}

Vector * MWD(Vector * wf,  double f, uint32_t M, uint32_t L){
  return MovingAverage(
      OffsetDifferentiate(Deconvolute(wf, f), M), L);
}
