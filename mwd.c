#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "algo.h"

#define NdataMax 1000 * 1000

Vector * ReadWF(const char * filename);

int main(int argc, char *argv[]) {
  Vector * wf = ReadWF("samples/purdue_full_wf0.csv");

  Vector * mwd = MWD(wf, 0.999993, 6000, 600);
  for (uint32_t i = 0; i < mwd->size; ++i) {
    /* printf("%.9lf\n", mwd->data[i]); */
  }

  VectorFree(mwd);
  VectorFree(wf);
  return 0;
}

Vector * ReadWF(const char *filename){
  Vector * wf0 = VectorInit();
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Cannot open file.\n");
    exit(1);
  }

  double val0, val1;
  while (fscanf(fp, "%lf,%lf", &val0, &val1) == 2) {
    VectorAppend(wf0, val1);
  }

  fclose(fp);
  return wf0;
}

