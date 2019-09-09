#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

#include "algo.h"
#include "vector.h"

Vector * ReadWF(const char * filename);
long int getMicrotime(){
  struct timeval currentTime;
  gettimeofday(&currentTime, NULL);
  return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}

int main(int argc, char *argv[]) {
  long int start = getMicrotime();
  Vector * wf = ReadWF("samples/purdue_full_wf0.csv");
  long int stop = getMicrotime();
  printf("Reading time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  start = getMicrotime();
  Vector * mwd = MWD(wf, 0.999993, 6000, 600);
  stop = getMicrotime();
  printf("MWD time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);
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
