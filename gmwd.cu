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

Vector * ReadWF(const char * filename);
long int getMicrotime();

#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) 
      exit(code);
  }
}

template <typename T>
struct minus_and_divide : public thrust::binary_function<T,T,T> {
    T w;
    minus_and_divide(T w) : w(w) {}
    __host__ __device__
    T operator()(const T& a, const T& b) const {
        return (a - b) / w;
    }
};

template <typename InputVector, typename OutputVector>
void simple_moving_average(const InputVector& data, size_t w, OutputVector& output) {
    typedef typename InputVector::value_type T;
    if (data.size() < w)
        return;
    // allocate storage for cumulative sum
    thrust::device_vector<T> temp(data.size() + 1);
    // compute cumulative sum
    thrust::exclusive_scan(data.begin(), data.end(), temp.begin());
    temp[data.size()] = data.back() + temp[data.size() - 1];
    // compute moving averages from cumulative sum
    thrust::transform(temp.begin() + w, temp.end(), temp.begin(), output.begin(), minus_and_divide<T>(T(w)));
}

__global__ void gpuAdd(double *a, double *b, double *c, uint32_t n);
__global__ void gpuSubtract(double *a, double *b, double *c, uint32_t n);
__global__ void gpuScale(double *a, double *b, double f, uint32_t n);
__global__ void gpuMovingAverage(double *a, double *b, uint32_t window, uint32_t n);

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
  Vector * mwd = MWD(hostWf0, f, M, L);
  stop = getMicrotime();
  printf("CPU MWD time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  // GPU things, still need to precompute the deconvolution

  printf("nSamples,pre,copy0,gpu,copy1,post\n");
  Benchmark(hostWf0, hostWf0->size, 100, f, M, L);
  Benchmark(hostWf0, 100000, 100, f, M, L);
  Benchmark(hostWf0, 50000, 50, f, M, L);
  Benchmark(hostWf0, 25000, 30, f, M, L);
  Benchmark(hostWf0, 10000, 20, f, M, L);
  Benchmark(hostWf0, 5000, 20, f, M, L);
  Benchmark(hostWf0, 2500, 20, f, M, L);
  // done
  VectorFree(mwd);
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


__global__ void gpuScale(double *a, double *b, double f, uint32_t n) {
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if (gId < n)
    b[gId] = f * a[gId];
}

__global__ void gpuAdd(double *a, double *b, double *c, uint32_t n) {
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if (gId < n)
    c[gId] = a[gId] + b[gId];
}

__global__ void gpuSubtract(double *a, double *b, double *c, uint32_t n) {
  uint32_t gId = blockIdx.x*blockDim.x + threadIdx.x; // global id
  if (gId < n)
    c[gId] = a[gId] - b[gId];
}
__global__ void gpuMovingAverage(double *a, double *b, uint32_t window, uint32_t n) {
}

void Benchmark(Vector * wf, uint32_t nSamples, uint32_t nLoops, double f,
    uint32_t M, uint32_t L){
  uint32_t nChunks = wf->size / nSamples;

  for (int j = 0; j < nLoops; j++) {
    for (int i = 0; i < nChunks; i++) {
      uint32_t nBytes = nSamples * sizeof(double);
      Vector * sWf = VectorInit();
      VectorCopy(sWf, wf, i * nSamples, nSamples);

      // GPU things, still need to precompute the deconvolution
      cudaEvent_t		time1, time2, time3, time4;
      float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime; // GPU code run times

      thrust::device_vector<double> devMWD(nSamples);

      int blockSize, gridSize;
      blockSize = 1024;
      gridSize = (int) ceil((float)nBytes / blockSize);

      cudaEventCreate(&time1);
      cudaEventCreate(&time2);
      cudaEventCreate(&time3);
      cudaEventCreate(&time4);

      long int start = getMicrotime();
      Vector * hostDeconv = Deconvolute(sWf, f);
      long int stop = getMicrotime();
      long int precomputeTime = stop - start;

      cudaEventRecord(time1, 0);
      thrust::device_vector<double> devDeconv(hostDeconv->data, hostDeconv->data + nSamples);
      thrust::device_vector<double> devOdiff(nSamples - M);
      cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done

      // start calculating
      gpuSubtract<<<gridSize, blockSize>>>(
          thrust::raw_pointer_cast(devDeconv.data() + M),
          thrust::raw_pointer_cast(devDeconv.data()),
          thrust::raw_pointer_cast(devOdiff.data()), nSamples - M);
      thrust::device_vector<double> devMA(devOdiff.size() - L - 1);
      simple_moving_average(devOdiff, L, devMA);

      gpuErrChk(cudaDeviceSynchronize());
      cudaEventRecord(time3, 0);
      thrust::host_vector<double> hostDMW = devMA;
      cudaEventRecord(time4, 0);

      cudaEventSynchronize(time1);
      cudaEventSynchronize(time2);
      cudaEventSynchronize(time3);
      cudaEventSynchronize(time4);
      cudaEventElapsedTime(&totalTime, time1, time4);
      cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
      cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
      cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

      gpuErrChk(cudaDeviceSynchronize());

      /* start = getMicrotime(); */
      /* Vector * hostDMW = MovingAverage(hostOdiff1, L); */
      /* stop = getMicrotime(); */
      long int postcomputeTime = 0;

      /* printf("nSamples,pre,copy0,gpu,copy1,post"); */
      printf("%u,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f\n", nSamples,
          (float)precomputeTime/1000, tfrCPUtoGPU, kernelExecutionTime,
          tfrGPUtoCPU, (float)postcomputeTime/1000);
      // done
      VectorFree(sWf);
      /* VectorFree(hostDMW); */
      VectorFree(hostDeconv);
    }
  }
}
