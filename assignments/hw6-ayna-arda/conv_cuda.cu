/*
  MPI+CUDA
  CS455 GPU Cluster Programming
  HW6 Convolution
  11/1/2025
  Andrew Sohn
 */

#include <iostream>
using std::cerr;
using std::endl;

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <unistd.h>
#include <sys/time.h>

extern "C" {
  int conv_dev(int nprocs, int my_rank, int my_work, int *in_image,int *out_image, int height, int width, int filter_dim,int *filter_cpu);
}

//#define THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK 256

#define TILE_WIDTH 4
#define MAX_TILE_WIDTH 16
#define MAX_TILE MAX_TILE_WIDTH
#define FILTER_DIM 3
#define FILTER_RADIUS 1
#define MAX_MASK_DIM 5

//__constant__ int filter_dev[FILTER_DIM][FILTER_DIM];
__constant__ int filter_dev[FILTER_DIM*FILTER_DIM];

void conv_host_cpu(int my_rank, int my_work, int* input, int* output, unsigned int height, unsigned int width, int *filter_cpu) {
  int out_row,out_col,sum = 0;
  int filter_row,filter_col,in_row,in_col;

  int offset = my_rank*my_work;
  int start_row = offset, end_row=offset+my_work;
  int cnt=0;

  for(out_row=start_row; out_row<end_row; out_row++) { /* start at offset */
    for(out_col=0; out_col<width; out_col++) {
      sum = 0;
      for (filter_row=0; filter_row<FILTER_DIM; filter_row++) {
        for (filter_col=0; filter_col<FILTER_DIM; filter_col++) {
          in_row = out_row + filter_row - FILTER_RADIUS;
          in_col = out_col + filter_col - FILTER_RADIUS;
          if (in_row >= 0 && in_row < static_cast<int>(height) &&
              in_col >= 0 && in_col < static_cast<int>(width)) {
            sum += input[in_row*width + in_col] *
                   filter_cpu[filter_row*FILTER_DIM + filter_col];
          }
        }
      }
      output[cnt++] = sum;
    }
  }
}

// both host and dev have values at the base address
int compare_cpu(int my_rank, int my_work, int *host, int *dev, int height, int width) {
  int i,j,idx,flag=1;

  for (i=0; i<my_work; i++) {
    for (j=0; j<width; j++) {
      idx = i*width + j;
      if (dev[idx] != host[idx]) {
	printf("DIFFERENT: rank=%d: dev[%d][%d]=%d != host[%d][%d]=%d\n", \
	       my_rank,i,j,dev[idx],i,j,host[idx]);
	flag = 0;
	return flag;
      }
    }
  }

  return flag;
}


__global__ void conv_dev_cuda(int my_rank,int my_work, int height,int width, int *input, int *output){
  int sum = 0;
  int filter_row,filter_col,in_row,in_col;

  // start at row: my_rank*height/nprocs*width 
  int out_row = blockIdx.y * blockDim.y + threadIdx.y + my_rank*my_work;
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;

  if(out_row < height && out_row < (my_rank+1)*my_work && out_col < width) {
    for (filter_row=0; filter_row<FILTER_DIM; filter_row++) {
      for (filter_col=0; filter_col<FILTER_DIM; filter_col++) {
        in_row = out_row + filter_row - FILTER_RADIUS;
        in_col = out_col + filter_col - FILTER_RADIUS;
        if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
          sum += input[in_row*width + in_col] *
                 filter_dev[filter_row*FILTER_DIM + filter_col];
        }
      }
    }
    int local_row = out_row - my_rank*my_work;
    output[local_row*width + out_col] = sum;
  }
}

void print_lst_cpu(int name,int rank,int n, int *l){
  int i=0;
  printf("CPU rank=%d: %d: size=%d:: ",rank,name,n);
  for (i=0; i<n; i++) printf("%x ",l[i]);
  printf("\n");
}

void print_filter_cpu(int name,int rank,int n, int *buf){
  int i=0,j;
  printf("CPU rank=%d: %d: size=%d:: ",rank,name,n);
  for (i=0; i<n; i++) 
    for (j=0; j<n; j++) printf("%x ",*buf++); /* buf[i][j]); */
  printf("\n");
}

void init_filter(int *buf) {
  int i,j,cnt=0;
  for (i=0; i<FILTER_DIM; i++) 
    for (j=0; j<FILTER_DIM; j++) *buf++ = cnt++;
}

int conv_dev(int nprocs, int my_rank, int my_work, int *h_in_image,int *h_out_image, int height, int width, int filter_dim,int *filter_cpu) {
  int in_size=0,out_size=0,filter_size_bytes=0;

  int *d_in_image, *d_out_image, *h_out_image_cpu=0;
  int gx_dim,gy_dim,bx_dim,by_dim;

  struct timeval timecheck;
  cudaStream_t stream;

  in_size = height * width;
  out_size = in_size;
  filter_size_bytes = filter_dim*filter_dim*sizeof(int);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,0);
  if (my_rank == 0) {
  printf("\n**** properties: rank=%d *****\n",my_rank);
  printf("prop.name=%s\n", prop.name);
  printf("prop.multiProcessorCount=%d\n", prop.multiProcessorCount);
  printf("prop.major=%d minor=%d\n", prop.major, prop.minor);
  printf("prop.maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
  printf("maxThreadsDim.x=%d maxThreadsDim.y=%d maxThreadsDim.z=%d\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
  printf("prop.maxGridSize.x=%d maxGridSize.y=%d maxGridSize.z=%d\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
  printf("prop.maxThreadsPerMultiProcessor=%d\n", prop.maxThreadsPerMultiProcessor);
  printf("prop.totalGlobalMem=%u\n", prop.totalGlobalMem);
  printf("prop.regsPerBlock=%d\n", prop.regsPerBlock);
  printf("**** properties: rank=%d *****\n",my_rank);
  printf("\n");
  }

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  checkCudaErrors(cudaEventRecord(start, stream));

  /* copy filter_cpu to filter_dev */
  cudaMemcpyToSymbolAsync(filter_dev, filter_cpu, filter_size_bytes, 0, cudaMemcpyHostToDevice, stream);

  unsigned int in_bytes = sizeof(int) * in_size;
  unsigned int out_bytes = sizeof(int) * out_size;
  unsigned int my_work_bytes = sizeof(int) * my_work*width;

  h_out_image_cpu = (int *) malloc(sizeof(int) * out_size);

  printf("rank=%d: input image:(height=%d x width=%d) = in_bytes=%d => output image:(height=%d x width=%d) = out_bytes=%d in bytes\n",
	 my_rank,height,width,in_bytes,height,width,out_bytes);

  long dev_start, dev_end, dev_elapsed;
  gettimeofday(&timecheck, NULL);
  dev_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;

  cudaMalloc(reinterpret_cast<void **>(&d_in_image), in_bytes);
  cudaMalloc(reinterpret_cast<void **>(&d_out_image), out_bytes);

  // copy host memory to device
  cudaMemcpyAsync(d_in_image, h_in_image, in_bytes, cudaMemcpyHostToDevice, stream);

  bx_dim = by_dim = 4;
  /*CHANGE: Round up in case there is a remainder.*/
  gx_dim = (width + bx_dim - 1)/bx_dim;
  gy_dim = (my_work + by_dim - 1)/by_dim;
  
  dim3 grid(gx_dim,gy_dim);
  dim3 threads(bx_dim,by_dim);	/* by_dim,bx_dim gy_dim,gx_dim */
  printf("rank=%d: CUDA kernel launch: grid(%d,%d), block(%d,%d)\n", my_rank, gx_dim, gy_dim, bx_dim,by_dim);

  /* Call cuda function */
  conv_dev_cuda<<<grid, threads, 0, stream>>>(my_rank, my_work, height, width, d_in_image, d_out_image);

  // copy to the base address
  cudaMemcpyAsync(h_out_image, d_out_image, my_work_bytes, cudaMemcpyDeviceToHost, stream);

  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  gettimeofday(&timecheck, NULL);
  dev_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  dev_elapsed = dev_end - dev_start;
  
  printf("dev time: rank=%d: %d procs: %ld msecs\n",
	   my_rank, nprocs, dev_elapsed);
  printf("CPU %d: CUDA time=%.3f msec\n",my_rank,msecTotal);

  fflush(stdout);

  // Compute on the cpu
  conv_host_cpu(my_rank, my_work, h_in_image, h_out_image_cpu,
                height, width, filter_cpu);

  if (compare_cpu(my_rank,my_work,h_out_image_cpu,h_out_image,height,width)) /* h_C is from dev */
    printf("\nCPU %d: Test: PASS: host == dev\n\n", my_rank);
  else
    printf("\nCPU %d: Test: FAIL: host != dev\n\n", my_rank);

  fflush(stdout);
  cudaFree(d_in_image);
  cudaFree(d_out_image);
  free(h_out_image_cpu);
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
  return 0;
}
