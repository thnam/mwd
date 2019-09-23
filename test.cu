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

#include "prefixScan.h"
#include "gpuAlgo.h"

extern "C"{
#include "algo.h"
#include "vector.h"
}

Vector * ReadWF(const char * filename);
long int getMicrotime();

#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))

void Benchmark(Vector * wf, uint32_t nSamples, uint32_t nLoops, double f,
    uint32_t M, uint32_t L);

int main(int argc, char *argv[]) {
  long int start = getMicrotime();
  Vector * hostWf0 = ReadWF("samples/purdue_full_wf0.csv");
  long int stop = getMicrotime();
  printf("Reading time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  // default params
  double f = 0.999993;
  uint32_t M = 400;
  uint32_t L = 200;

  // CPU serial caculation
  start = getMicrotime();
  Vector * deconv = Deconvolute(hostWf0, f);
  Vector * odiff = OffsetDifferentiate(deconv, M);
  Vector * mavg = MovingAverage(odiff, L);
  stop = getMicrotime();
  printf("CPU MWD time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  // GPU
  Benchmark(hostWf0, hostWf0->size, 1, f, M, L);

  // done
  VectorFree(mavg);
  VectorFree(deconv);
  VectorFree(odiff);
  VectorFree(hostWf0);
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


void Benchmark(Vector * origWaveform, uint32_t nSamples, uint32_t nLoops,
    double f, uint32_t M, uint32_t L){
  uint32_t nChunks = origWaveform->size / nSamples;

  GpuTimer gTimer;

  for (int j = 0; j < nLoops; j++) {
    for (int i = 0; i < nChunks; i++) {
      uint32_t nBytes = nSamples * sizeof(double);
      Vector * subWaveform = VectorInit();
      VectorCopy(subWaveform, origWaveform, i * nSamples, nSamples);

      // CPU result, for verification
      Vector * deconv = Deconvolute(subWaveform, f);
      Vector * odiff = OffsetDifferentiate(deconv, M);
      Vector * mavg = MovingAverage(odiff, L);

      uint32_t blockSize = 1024;
      uint32_t gridSize = (int) ceil((float)nBytes / blockSize);

      // prepare device data
      double * devInput;
      double * devScanSum;
      double * devDeconv;
      double * devODiff;
      double * devMWD;
      checkCudaErrors(cudaMalloc(&devInput, nBytes));
      checkCudaErrors(cudaMalloc(&devScanSum, nBytes));
      checkCudaErrors(cudaMalloc(&devDeconv, nBytes));
      checkCudaErrors(cudaMalloc(&devODiff, nBytes));
      checkCudaErrors(cudaMalloc(&devMWD, nBytes));

      gTimer.Start();
      checkCudaErrors(cudaMemcpy(devInput, subWaveform->data, nBytes, cudaMemcpyHostToDevice));
      gTimer.Stop();
      std::cout << "Host to dev: " << gTimer.Elapsed() << std::endl;

      // 1st step: prefixScan
      gTimer.Start();
      sum_scan_blelloch(devScanSum, devInput, nSamples);
      gpuDeconvolute<<<gridSize, blockSize>>>(devInput, devScanSum, devDeconv, f, nSamples);
      gTimer.Stop();
      std::cout << "Deconv: " << gTimer.Elapsed() << std::endl;

      gTimer.Start();
      gpuOffsetDifferentiate<<<gridSize, blockSize>>>(devDeconv, devODiff, M, nSamples);
      gTimer.Stop();
      std::cout << "Odiff: " << gTimer.Elapsed() << std::endl;

      gTimer.Start();
      gpuMovingAverage<<<gridSize, blockSize>>>(devODiff, devMWD, L, nSamples);
      gTimer.Stop();
      std::cout << "Moving average: " << gTimer.Elapsed() << std::endl;

      double * hostScanSum = new double[nBytes];
      double * hostDeconv = new double[nBytes]; 
      double * hostODiff = new double[nBytes];
      double * hostMWD = new double[nBytes];
      gTimer.Start();
      checkCudaErrors(cudaMemcpy(hostMWD, devMWD, nBytes, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hostScanSum, devScanSum, nBytes, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hostDeconv, devDeconv, nBytes, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hostODiff, devODiff, nBytes, cudaMemcpyDeviceToHost));
      gTimer.Stop();
      std::cout << "Dev to host: " << gTimer.Elapsed() << std::endl;

      // 
      bool match = true;
      double tolerance = 0.0001;
      for (uint32_t i = 0; i < nSamples; ++i) {
        if ((hostDeconv[i] - deconv->data[i]) > tolerance) {
          printf("Deconv mismatch at %ud\n", i);
          match = false;
          break;
        }
        if ((hostODiff[i] - odiff->data[i]) > tolerance) {
          printf("Odiff mismatch at %ud\n", i);
          match = false;
          break;
        }
        if ((hostMWD[i] - mavg->data[i]) > tolerance) {
          printf("Mavg mismatch at %ud\n", i);
          match = false;
          break;
        }
      }
      printf("All matched!\n");

      printf("i: orig scansum  deconv odiff mwd\n");
      for (int i = 0; i < 20; ++i) {
        printf("%010d: %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf\n",
            i, subWaveform->data[i], hostScanSum[i], hostDeconv[i], hostODiff[i],
            hostMWD[i], mavg->data[i]);
      }

      for (uint32_t i = 0; i < nSamples; ++i) {
        printf("%.12lf\n", hostMWD[i]);
      }
      // clean up
      VectorFree(subWaveform);
      VectorFree(mavg);
      VectorFree(deconv);
      VectorFree(odiff);
      checkCudaErrors(cudaFree(devScanSum));
      checkCudaErrors(cudaFree(devDeconv));
      checkCudaErrors(cudaFree(devODiff));
      checkCudaErrors(cudaFree(devMWD));
      checkCudaErrors(cudaFree(devInput));

      delete [] hostMWD;
      delete [] hostDeconv;
      delete [] hostODiff;
      delete [] hostScanSum;
    }
  }
}
