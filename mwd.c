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

  // big chunk
  uint32_t nLoops = 1000;
  long int avgMWDTime = 0;
  Vector * mwd = NULL;
  double f = 0.999993;
  uint32_t M = 6000;
  uint32_t L = 600;

  for (uint32_t i = 0; i < nLoops; ++i) {
    start = getMicrotime();
    mwd = MWD(wf, f, M, L);
    stop = getMicrotime();
    avgMWDTime += (stop - start);
  }

  avgMWDTime /= nLoops;
  printf("chunkSize %u", wf->size);
  printf(", MWD time %lu usec = %.2f ms\n", avgMWDTime, (float)avgMWDTime/1000);

  // small chunk
  uint32_t chunkSize = 2500;
  uint32_t nChunks = wf->size / chunkSize;
  M = 500;
  L = 200;
  nLoops = 100;
  avgMWDTime = 0;
  Vector * sMwd = VectorInit();
  Vector * sWf = VectorInit();
  for (int j = 0; j < nLoops; ++j) {
    for (uint32_t i = 0; i < nChunks; ++i) {
      VectorCopy(sWf, wf, i * chunkSize, chunkSize);
      start = getMicrotime();
      sMwd = MWD(sWf, f, M, L);
      stop = getMicrotime();
      avgMWDTime += stop - start;
    }
  }
  avgMWDTime /= nLoops * nChunks;
  printf("chunkSize %u", chunkSize);
  printf(", MWD time %lu usec = %.2f ms\n", avgMWDTime, (float)avgMWDTime/1000);
  
  //
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
