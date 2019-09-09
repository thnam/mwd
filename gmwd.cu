#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

extern "C"{
#include "algo.h"
#include "vector.h"
}

Vector * ReadWF(const char * filename);
long int getMicrotime();

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))

static void HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}

__global__ void gpuAdd(double *a, double *b, double *c, uint32_t n) {
  uint32_t gId = blockIdx.x*blockDim.x+threadIdx.x; // global id
  if (gId < n)
    c[gId] = a[gId] + b[gId];
}

__global__ void gpuMultiply(double *a, double *b, double f, uint32_t n) {
  uint32_t gId = blockIdx.x*blockDim.x+threadIdx.x; // global id
  if (gId < n)
    b[gId] = f * a[gId];
}

__global__ void gpuMovingAverage(double *a, double *b, uint32_t window,
    uint32_t n) {
}

int main(int argc, char *argv[]) {
  long int start = getMicrotime();
  Vector * wf = ReadWF("samples/purdue_full_wf0.csv");
  long int stop = getMicrotime();
  printf("Reading time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  double f = 0.999993;
  uint32_t M = 6000;
  uint32_t L = 600;

  start = getMicrotime();
  Vector * mwd = MWD(wf, f, M, L);
  stop = getMicrotime();
  printf("CPU MWD time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  // GPU things
	cudaEvent_t		time1, time2, time3, time4;
	float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime; // GPU code run times
	cudaError_t		cudaStatus;
  cudaDeviceProp	GPUprop;
  uint32_t nBytes = wf->size * sizeof(double);
  /* double * hostWf0 = wf->data; */
  double * devWf0;
  double * devMWD;
  double * hostMWD = (double *) malloc(nBytes);
  char SupportedBlocks[100];

  int blockSize, gridSize;
  blockSize = 1024;
  gridSize = (int) ceil((float)nBytes / blockSize);
  printf("blockSize %d, gridSize %d\n", blockSize, gridSize);

  cudaEventCreate(&time1);
  cudaEventCreate(&time2);
  cudaEventCreate(&time3);
  cudaEventCreate(&time4);

  cudaEventRecord(time1, 0);
	int NumGPUs = 0;
	cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		printf("\nNo CUDA Device is available\n");
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(EXIT_FAILURE);
	}
	cudaGetDeviceProperties(&GPUprop, 0);
	uint32_t SupportedKBlocks = (uint32_t)GPUprop.maxGridSize[0]
    * (uint32_t)GPUprop.maxGridSize[1] * (uint32_t)GPUprop.maxGridSize[2] / 1024;
	uint32_t SupportedMBlocks = SupportedKBlocks / 1024;
	sprintf(SupportedBlocks, "%u %c",
      (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks,
      (SupportedMBlocks >= 5) ? 'M' : 'K');
	uint32_t MaxThrPerBlk = (uint32_t)GPUprop.maxThreadsPerBlock;

	printf("--------------------------------------------------------------------------\n");
	printf("%s    ComputeCapab=%d.%d  [max %s blocks; %d thr/blk] \n", 
			GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
	printf("--------------------------------------------------------------------------\n");

  HANDLE_ERROR(cudaMalloc((void **) &devWf0, nBytes));
  HANDLE_ERROR(cudaMalloc((void **) &devMWD, nBytes));

	HANDLE_ERROR(cudaMemcpy(devWf0, wf->data, nBytes, cudaMemcpyHostToDevice));

	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
  gpuMultiply<<<gridSize, blockSize>>>(devWf0, devMWD, f, wf->size);
  cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
        "\ncudaDeviceSynchronize returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(time3, 0);
  HANDLE_ERROR(cudaMemcpy(hostMWD, devMWD, nBytes, cudaMemcpyDeviceToHost));
  cudaEventRecord(time4, 0);

	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);
	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

	cudaStatus = cudaDeviceSynchronize();
  /* checkError(cudaGetLastError());	// screen for errors in kernel launches */
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		exit(EXIT_FAILURE);
	}

  uint32_t i = 0;
  for (i = 0; i < wf->size; ++i) {
    printf("%lf %lf\n", wf->data[i], hostMWD[i]);
  }

	printf("CPU->GPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n",
      tfrCPUtoGPU, DATAMB(nBytes), DATABW(nBytes, tfrCPUtoGPU));
	printf("Kernel Execution    =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n",
      kernelExecutionTime, DATAMB(2*nBytes), DATABW(2*nBytes, kernelExecutionTime));
	printf("GPU->CPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n",
      tfrGPUtoCPU, DATAMB(nBytes), DATABW(nBytes, tfrGPUtoCPU));
	printf("--------------------------------------------------------------------------\n");
	printf("Total time elapsed  =%7.2f ms       %4d MB  ...  %6.2f GB/s\n",
      totalTime, DATAMB((2 * nBytes + 2*nBytes)),
      DATABW((2 * nBytes + 2*nBytes), totalTime));
  printf("--------------------------------------------------------------------------\n\n");


  // done
  cudaFree(devWf0);
  cudaFree(devMWD);
  free(hostMWD);
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

long int getMicrotime(){
  struct timeval currentTime;
  gettimeofday(&currentTime, NULL);
  return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}
