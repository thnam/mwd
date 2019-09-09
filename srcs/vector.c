#include "vector.h"
#include<stdlib.h>
#include<stdio.h>
#include<string.h>

Vector * VectorInit(){
  Vector * A = (Vector*)malloc(sizeof(Vector)); 
  A->size = 0;
  A->capacity = VECTOR_INITIAL_CAPACITY;
  A->data = malloc(sizeof(double) * VECTOR_INITIAL_CAPACITY);

  return A;
}


void VectorAppend(Vector *vector, double value){
  VectorExpandIfFull(vector);
  vector->data[vector->size++] = value;
}

double VectorGet(Vector *vector, uint32_t index){
  return 0;
}

void VectorSet(Vector *vector, uint32_t index, double value){}
void VectorExpandIfFull(Vector *vector){
  if (vector->size >= vector->capacity) {
    vector->capacity = (int)(VECTOR_GROWING_FACTOR * vector->capacity);
    vector->data = realloc(vector->data, sizeof(double) * vector->capacity);
  }
}

void VectorExpand(Vector *vector){
    vector->capacity = (int)(VECTOR_GROWING_FACTOR * vector->capacity);
    vector->data = realloc(vector->data, sizeof(double) * vector->capacity);
}

void VectorFree(Vector *vector){
  free(vector->data);
  free(vector);
}

void VectorCopy(Vector *vd, Vector *vs, uint32_t start, uint32_t stop){
  if ((start >= stop)|| (stop > vs->size) ) {
    return;
  }
  
  while (vd->capacity < (stop - start)){
    VectorExpand(vd);
  }

  memcpy(vd->data, vs->data + start, (stop - start) * sizeof(double));
  vd->size = stop - start;
}
