#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "algo.h"
#include "vector.h"

Vector * ReadWF(const char * filename);
long int getMicrotime();
void Benchmark(Vector * data, uint32_t chunkSize, uint32_t nLoops, double f, uint32_t M, uint32_t L);
void Benchmark(Vector * data, uint32_t chunkSize, uint32_t nLoops, double f, uint32_t M, uint32_t L){
  uint32_t nChunks = data->size / chunkSize;
  long int * elapse = (long int *)malloc(sizeof(long int) * nLoops * nChunks);

  Vector * sMwd = VectorInit();
  Vector * sWf = VectorInit();
  for (uint32_t j = 0; j < nLoops; ++j) {
    for (uint32_t i = 0; i < nChunks; ++i) {
      VectorCopy(sWf, data, i * chunkSize, chunkSize);
      long int start = getMicrotime();
      sMwd = MWD(sWf, f, M, L);
      long int stop = getMicrotime();
      elapse[j * nChunks + i] = stop - start;
    }
  }

  double sum = 0.0;
  double mean;
  double stdDev = 0.0;
  for (int i = 0; i < nLoops * nChunks; ++i) {
    sum += elapse[i];
  }
  mean = sum/nChunks/nLoops;
  for (int i = 0; i < nLoops * nChunks; ++i) {
    stdDev += pow(elapse[i] - mean, 2);
  }
  stdDev = sqrt(stdDev/(nLoops * nChunks));

  /* printf("Chunk size %u, avg time %.1f +/- %.1f us = %.3f +/-  %.3f\n",  */
      /* chunkSize, mean, stdDev, mean/1000, stdDev/1000); */

  printf("%u,%.1f,%.1f,%.4f,%.4f\n",
      chunkSize, mean, stdDev, mean/chunkSize, stdDev/chunkSize);

  free(elapse);
  VectorFree(sMwd);
  VectorFree(sWf);
}

int main(int argc, char *argv[]) {
  long int start = getMicrotime();
  Vector * wf = ReadWF("samples/purdue_full_wf0.csv");
  long int stop = getMicrotime();
  printf("Reading time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  // big chunk
  /* uint32_t nLoops = 1000; */
  /* long int avgMWDTime = 0; */
  /* Vector * mwd = NULL; */

  /* for (uint32_t i = 0; i < nLoops; ++i) { */
    /* start = getMicrotime(); */
    /* mwd = MWD(wf, f, M, L); */
    /* stop = getMicrotime(); */
    /* avgMWDTime += (stop - start); */
  /* } */

  /* avgMWDTime /= nLoops; */
  /* printf("chunkSize %u", wf->size); */
  /* printf(", MWD time %lu usec = %.2f ms\n", avgMWDTime, (float)avgMWDTime/1000); */

  // small chunk
  double f = 0.999993;
  uint32_t M = 100;
  uint32_t L = 100;
  printf("size,mean,stdDev,normMean,normStdDev\n");
  Benchmark(wf, 300, 50, f, M, L);
  Benchmark(wf, 500, 70, f, M, L);

  f = 0.999993;
  M = 400;
  L = 200;
  Benchmark(wf, 1000, 70, f, M, L);
  Benchmark(wf, 2500, 100, f, M, L);
  Benchmark(wf, 5000, 200, f, M, L);
  Benchmark(wf, 10000, 400, f, M, L);
  Benchmark(wf, 20000, 800, f, M, L);
  Benchmark(wf, 50000, 1200, f, M, L);
  Benchmark(wf, 100000, 2000, f, M, L);
  Benchmark(wf, 250000, 2000, f, M, L);

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

long int getMicrotime(){
  struct timeval currentTime;
  gettimeofday(&currentTime, NULL);
  return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}
