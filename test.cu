#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/random.h>

#include <thrust/device_vector.h>
extern "C"{
#include "algo.h"
#include "vector.h"
}
#include "scan.h"

Vector * ReadWF(const char * filename);
long int getMicrotime();
void cpu_sum_scan(double* const h_out, const double* const h_in,
    const size_t numElems);

int main(int argc, char *argv[]) {
  long int start = getMicrotime();
  Vector * hostWf0 = ReadWF("samples/purdue_full_wf0.csv");
  long int stop = getMicrotime();
  printf("Reading time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  /* uint32_t h_in_len = hostWf0->size; */
  uint32_t h_in_len = 1 << 11;

  double* h_out_blelloch = new double[h_in_len];
  double* d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(double) * h_in_len));
  checkCudaErrors(cudaMemcpy(d_in, hostWf0->data, sizeof(double) * h_in_len,
        cudaMemcpyHostToDevice));
  double* d_out_blelloch;
  checkCudaErrors(cudaMalloc(&d_out_blelloch, sizeof(double) * h_in_len));

  start = std::clock();
  sum_scan_blelloch(d_out_blelloch, d_in, h_in_len);
  double  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  std::cout << "GPU time: " << duration << std::endl;
  checkCudaErrors(cudaMemcpy(h_out_blelloch, d_out_blelloch,
        sizeof(double) * h_in_len, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_out_blelloch));

  double* h_out_naive = new double[h_in_len];
  start = std::clock();
  cpu_sum_scan(h_out_naive, hostWf0->data, h_in_len);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  std::cout << "CPU time: " << duration << std::endl;
  bool match = true;
  int index_diff = 0;
  for (int i = 0; i < h_in_len; ++i)
  {
    if (h_out_naive[i] != h_out_blelloch[i])
    {
      match = false;
      index_diff = i;
      break;
    }
  }
  std::cout << "Match: " << match << std::endl;

  /* for (uint32_t i = 0; i < h_in_len; ++i) { */
  for (uint32_t i = 0; i < 10; ++i) {
    printf("%.9lf %.9lf %.9lf\n", hostWf0->data[i],
        h_out_naive[i], h_out_blelloch[i]);
  }

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

void cpu_sum_scan(double* const h_out, const double* const h_in,
    const size_t numElems)
{
  double run_sum = 0.;
  for (int i = 0; i < numElems; ++i)
  {
    h_out[i] = run_sum;
    run_sum = run_sum + h_in[i];
  }
}
