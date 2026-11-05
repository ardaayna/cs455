/*
  Andrew Sohn
  10/27/2025
  CS455 MPI+CUDA Programming
  *
  The MPI+CUDA program compiles and passes the test because they are all zeros.
  Fill the functions.

  NOTE:
  need to place nvidia Common directory two dirs above the current dir
  or 
  change the Makefile reference of Common
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

#define TILE_WIDTH 4
#define MAX_TILE_WIDTH 16
#define THREADS_PER_BLOCK 256

extern "C" {
  int matrix_multiply_cuda(int nprocs, int my_rank,int n, int my_work,int *h_A,int *h_B,int *h_C,int gx_dim,int gy_dim,int bx_dim,int by_dim );
}

int matrix_multiply_cpu(int my_rank,int *a, int *b, int *c, int n, int my_work) {
  /* 
     int i, j, k, sum=0;

     Fill function

 */
    for (int i = 0; i < my_work; i++) {
        for (int k = 0; k < n; k++) {
            int sum = 0;
            for (int j = 0; j < n; j++) {
                sum += a[i*n + j] * b[j*n + k];
            }
            c[i*n + k] = sum;
        }
    }

    return 0;
}

int compare_cpu(int my_rank, int *host, int *dev, int n, int my_work) {
  int i,j,idx;

  for (i=0; i<my_work; i++) {
    for (j=0; j<n; j++) {
      //row length is n, not my_work (original code used my_work)
      idx = i * n + j;
      if (dev[idx] != host[idx]) {
	printf("DIFFERENT: rank=%d: dev[%d][%d]=%d != host[%d][%d]=%d\n", \
	       my_rank,i,j,dev[idx],i,j,host[idx]);
	return 0;
      }
    }
  }

  return 1;
}

__global__ void mat_mult_cuda(int my_rank, int a_width,int my_work, int *d_a, int *d_b, int *d_c, int tile_width){
  /* 
  __shared__ int a_shared[][] ...
  __shared__ int b_shared[][] ...
 */

  /* Fill this func */
    __shared__ int a_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ int b_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * tile_width + ty;
    int col = blockIdx.x * tile_width + tx;

    int sum = 0;
    int num_tiles = (a_width + tile_width - 1) / tile_width;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * tile_width + tx;
        int b_row = t * tile_width + ty;

        if (row < my_work && a_col < a_width)
            a_shared[ty][tx] = d_a[row * a_width + a_col];
        else
            a_shared[ty][tx] = 0;

        if (b_row < a_width && col < a_width)
            b_shared[ty][tx] = d_b[b_row * a_width + col];
        else
            b_shared[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < tile_width; k++) {
            sum += a_shared[ty][k] * b_shared[k][tx];
        }

        __syncthreads();
    }

    if (row < my_work && col < a_width)
        d_c[row * a_width + col] = sum;

}

void print_lst_cpu(int name,int rank,int n, int *l){
  int i=0;
  printf("CPU rank=%d: %d: ",rank,name);
  for (i=0; i<n; i++) printf("%x ",l[i]);
  printf("\n");
}

int matrix_multiply_cuda(int nprocs, int my_rank,int n, int my_work,int *h_A,int *h_B,int *h_C,int gx_dim,int gy_dim,int bx_dim,int by_dim ) {
  int cuda_prod=0;
  int *d_A, *d_B, *d_C;
  struct timeval timecheck;

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

  unsigned int my_work_size = sizeof(int) * my_work * n;
  unsigned int mat_size = sizeof(int) * n * n;
  printf("rank=%d: my_work=%d data_size=%d bytes\n",my_rank,my_work,my_work_size);

  long dev_start, dev_end, dev_elapsed;
  gettimeofday(&timecheck, NULL);
  dev_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;

  int *h_C_on_cpu = (int *) malloc(my_work_size);

  cudaMalloc(reinterpret_cast<void **>(&d_A), my_work_size);
  cudaMalloc(reinterpret_cast<void **>(&d_B), mat_size);
  cudaMalloc(reinterpret_cast<void **>(&d_C), my_work_size);

  cudaMemcpy(d_A, h_A, my_work_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mat_size, cudaMemcpyHostToDevice);

  by_dim = bx_dim;
  //round up grid dimensions, same reason as .cpp
  gx_dim = (n + bx_dim - 1) / bx_dim;
  gy_dim = (my_work + by_dim - 1) / by_dim;

  dim3 grid(gx_dim,gy_dim);
  dim3 threads(bx_dim,by_dim);

  mat_mult_cuda<<<grid,threads>>>(my_rank,n,my_work,d_A, d_B, d_C,by_dim);

  cudaMemcpy(h_C,d_C,my_work_size, cudaMemcpyDeviceToHost);

  gettimeofday(&timecheck, NULL);
  dev_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  dev_elapsed = dev_end - dev_start;
  
  printf("dev time: rank=%d: %d procs: %ld msecs\n",
	   my_rank, nprocs, dev_elapsed);

  fflush(stdout);

  matrix_multiply_cpu(my_rank,h_A,h_B,h_C_on_cpu,n,my_work);

  if (compare_cpu(my_rank,h_C_on_cpu,h_C,n,my_work)) /* h_C is from dev */
    printf("\nrank=%d: Test CPU: PASS: host == dev\n", my_rank);
  else
    printf("\nrank=%d: Test CPU: FAIL: host != dev\n", my_rank);

  fflush(stdout);

  return cuda_prod;

}

/*************************************************
  End of file
*************************************************/
