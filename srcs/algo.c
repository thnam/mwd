#include "algo.h"
#include<stdlib.h>
#include<stdio.h>

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

Vector * VectorInit(){
  Vector * A = (Vector*)malloc(sizeof(Vector)); 
  A->size = 0;
  A->capacity = VECTOR_INITIAL_CAPACITY;
  A->data = malloc(sizeof(double) * VECTOR_INITIAL_CAPACITY);

  return A;
}


void VectorAppend(Vector *vector, double value){
  VectorExpand(vector);
  vector->data[vector->size++] = value;
}

double VectorGet(Vector *vector, uint32_t index){
  return 0;
}

void VectorSet(Vector *vector, uint32_t index, double value){}
void VectorExpand(Vector *vector){
  if (vector->size >= vector->capacity) {
    vector->capacity = (int)(VECTOR_GROWING_FACTOR * vector->capacity);
    /* vector = (Vector *) realloc(vector, 2 * sizeof(uint32_t) + sizeof(double) * vector->capacity); */
    vector->data = realloc(vector->data, sizeof(double) * vector->capacity);
  }
}

void VectorFree(Vector *vector){
  free(vector->data);
  free(vector);
}
