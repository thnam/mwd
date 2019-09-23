#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

#include <vector>
#include <string>
#include <map>

#include <thrust/device_vector.h>

#include "prefixScan.h"
#include "gpuAlgo.h"

extern "C"{
#include "algo.h"
#include "vector.h"
}

Vector * ReadWF(const char * filename);
long int getMicrotime();

std::map<std::string, std::vector<double>> timeCost; // msec

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
  uint32_t nLoops = 2000;
  uint32_t nSamples = hostWf0->size;

  while ((nSamples > 3 * (M + L)) && (nLoops > 0)){
    Benchmark(hostWf0, nSamples, nLoops, f, M, L);
    nSamples /= 2;
    nLoops /= 2;
  }
  M = 40;
  L = 20;
  while ((nSamples > 3 * (M + L)) && (nLoops > 0)){
    Benchmark(hostWf0, nSamples, nLoops, f, M, L);
    nSamples /= 2;
    nLoops /= 2;
  }

  for (auto e : timeCost)
    std::cout << e.first << ",";
  std::cout << "\n" << std::flush;

  uint32_t nTests = timeCost["cpu"].size();
  for (uint32_t i = 0; i < nTests; ++i) {
    for (auto e : timeCost){
      std::cout << e.second.at(i) <<",";
    }
    std::cout << "\n" << std::flush;
  }

  // done
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
      timeCost["samples"].push_back(nSamples);

      // CPU result, for verification
      long int start = getMicrotime();
      Vector * deconv = Deconvolute(subWaveform, f);
      Vector * odiff = OffsetDifferentiate(deconv, M);
      Vector * mavg = MovingAverage(odiff, L);
      long int stop = getMicrotime();
      timeCost["cpu"].push_back((double)(stop - start) / 1000);

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
      timeCost["cudaHost2Dev"].push_back(gTimer.Elapsed());

      // 1st step: prefixScan
      gTimer.Start();
      sum_scan_blelloch(devScanSum, devInput, nSamples);
      gpuDeconvolute<<<gridSize, blockSize>>>(devInput, devScanSum, devDeconv, f, nSamples);
      gTimer.Stop();
      timeCost["cudaDeconv"].push_back(gTimer.Elapsed());

      gTimer.Start();
      gpuOffsetDifferentiate<<<gridSize, blockSize>>>(devDeconv, devODiff, M, nSamples);
      gTimer.Stop();
      timeCost["cudaOdiff"].push_back(gTimer.Elapsed());

      gTimer.Start();
      gpuMovingAverage<<<gridSize, blockSize>>>(devODiff, devMWD, L, nSamples);
      gTimer.Stop();
      timeCost["cudaMavg"].push_back(gTimer.Elapsed());

      double * hostMWD = new double[nBytes];

      gTimer.Start();
      checkCudaErrors(cudaMemcpy(hostMWD, devMWD, nBytes, cudaMemcpyDeviceToHost));
      gTimer.Stop();
      timeCost["cudaDev2Host"].push_back(gTimer.Elapsed());

      // 
      double tolerance = 0.0001;
      for (uint32_t i = 0; i < nSamples; ++i) {
        if ((hostMWD[i] - mavg->data[i]) > tolerance) {
          printf("Mavg mismatch at %ud\n", i);
          break;
        }
      }
      /* printf("All matched!\n"); */

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
    }
  }
}
