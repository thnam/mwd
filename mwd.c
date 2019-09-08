#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "algo.h"

#define NsamplesMax 1000 * 1000

Waveform * ReadWF(const char * filename);

int main(int argc, char *argv[]) {
  Waveform * wf = ReadWF("samples/purdue_full_wf0.csv");

  Waveform * mwd = MWD(wf, 0.999993, 6000, 600);
  for (uint32_t i = 0; i < mwd->size; ++i) {
    printf("%.8lf\n", mwd->samples[i]);
  }

  return 0;
}

Waveform * ReadWF(const char *filename){

  Waveform * wf0 = (Waveform*) malloc(sizeof(uint32_t) + NsamplesMax * sizeof(double));
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Cannot open file.\n");
    exit(1);
  }

  double val0, val1;
  while (fscanf(fp, "%lf,%lf", &val0, &val1) == 2) {
    wf0->samples[wf0->size] = val1;
    wf0->size++;
  }
  return wf0;
}

