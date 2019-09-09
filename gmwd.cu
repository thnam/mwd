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

static void HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}

__global__ void vecAdd(double *a, double *b, double *c, int n) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  // Make sure we do not go out of bounds
  if (id < n)
    c[id] = a[id] + b[id];
}

__global__ void vecMultiply(double *a, double *b, double f, int n) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  // Make sure we do not go out of bounds
  if (id < n)
    b[id] = f * a[id];
}

int main(int argc, char *argv[]) {
  long int start = getMicrotime();
  Vector * wf = ReadWF("samples/purdue_full_wf0.csv");
  long int stop = getMicrotime();
  printf("Reading time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  start = getMicrotime();
  Vector * mwd = MWD(wf, 0.999993, 6000, 600);
  stop = getMicrotime();
  printf("CPU MWD time %ld usec = %ld ms\n", (stop - start), (stop - start)/1000);

  // GPU things
	cudaEvent_t		time1, time2, time3, time4;
	cudaError_t		cudaStatus, cudaStatus2;
	/* cudaDeviceProp	GPUprop; */
  uint32_t nBytes = wf->size * sizeof(double);
  /* double * hostWf0 = wf->data; */
  double * devWf0;
  double * devMWD;
  double * hostMWD = (double *) malloc(nBytes);
  double f = 0.999993;

  int blockSize, gridSize;
  blockSize = 1024;
  gridSize = (int) ceil((float)wf->size / blockSize);

  cudaEventCreate(&time1);
  cudaEventCreate(&time2);
  cudaEventCreate(&time3);
  cudaEventCreate(&time4);

  cudaEventRecord(time1, 0);
  cudaStatus = cudaMalloc((void **) &devWf0, wf->size);
  cudaStatus2 = cudaMalloc((void **) &devMWD, wf->size);
	if ((cudaStatus != cudaSuccess) || (cudaStatus2 != cudaSuccess)){
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpy(devWf0, wf->data, nBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
  vecMultiply<<<gridSize, blockSize>>>(devWf0, devMWD, f, wf->size);
  cudaEventRecord(time3, 0);
  cudaMemcpy(hostMWD, devMWD, nBytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(time4, 0);

  uint32_t i = 0;
  for (i = 0; i < wf->size; ++i) {
    printf("%lf\n", devMWD[i]);
  }

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
