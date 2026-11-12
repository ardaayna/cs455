/*
  MPI+CUDA Programming:
  Programming a cluster of CUDA-capable machines
  aka multi-node multi-GPU programming

  Andrew Sohn
  CS455/CS655 GPU Cluster Programming
  3/15/2025

  1-bit radix sorting for any number of bits

 */

#include <iostream>
using std::cerr;
using std::endl;

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <unistd.h>
#include <sys/time.h>

#define MASTER 0
#define ROOT 0
#define MAX_N 1<<12
#define MAX_ORDER 25
#define MAX_SIZE 1 << MAX_ORDER
#define MAX_BIN 1 << 8		/* get two hexa's */

#define ORG 1
#define LST 1
#define BUF 0
#define MAXBIT 32 

#define MAX_GRID_DIM 1<<12
#define MAX_BLOCK_DIM 1<<10
#define SH_GRID_DIM 1<<10		/* for shared mem declaration */
#define BLOCK_DIM 16

#define THREADS 1<<8

#define PASS 1
#define FAIL 0

extern int lst[MAX_SIZE], buf[MAX_SIZE], tmp[MAX_SIZE], copy[MAX_SIZE], cnt[MAX_SIZE], map[MAX_SIZE];
extern int bin[1<<MAX_BIN],cpu_bin[1<<MAX_BIN],host_bin[1<<MAX_BIN];
extern int source,radix;         /* 1 for LST 0 for BUF holding the current data */

extern void print_lst(const char *name,int rank,int n, int *l);

__device__ int d_blk_ctr=0;
int h_blk_ctr=0;

int sum(int rank, int size, int *data) {
  int accum = 0;
  for (int i = 0; i < size; i++) accum = accum + data[i];
  return accum;
}

void init_lst_with_flag(int *lst, int n, int mask,int flag) {
  srand(time(NULL));
  if (flag==0) 
    for (int i = 0; i < n; i++) *lst++ = 0;
  else if (flag==1) 
    for (int i = 0; i < n; i++) *lst++ = 1;
  else
    for (int i = 0; i < n; i++) *lst++ = rand() & mask; /* mask=0xffff or 0xff or 0xf ... */
}

__global__ void cnt_cuda_func(int, int iter, int n, int *, int *src, int *cnt, int *, int) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int i = bid*blockDim.x + tid;
  if (i < n) {
    // Flip the sign bit so the unsigned key order matches signed int order.
    unsigned int key = static_cast<unsigned int>(src[i]) ^ (1U << (MAXBIT - 1));
    cnt[i] = (key >> iter) & 1U;
  }
}

__global__ void mov_cuda_func(int, int, int n, int *dst, int *src, int *cnt, int *map, int, int ones_total) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int i = bid*blockDim.x + tid;

  if (i < n) {
    int bit = cnt[i];
    // Convert the inclusive scan to the number of ones before this element.
    int ones_left = map[i] - bit;
    int zeros_total = n - ones_total;
    int pos = bit ? zeros_total + ones_left : i - ones_left;
    dst[pos] = src[i];
  }
}

void inc_scan_cpu(int *prefix, int *lst, int n) {
  int i;
  prefix[0] = lst[0]; 
  for (i=1;i<n;i++) { 
    prefix[i] = prefix[i-1] + lst[i];
  }
}

int self_check(int n, int *lst) {
  int i,flag;
  flag = PASS;

  for (i=0;i<n-1;i++) {
    if (lst[i] > lst[i+1]) { flag = FAIL; break; }
  }
  if (flag == PASS) {
    printf("Check sorting: PASS\n");
  }else{
     printf("Check sorting: FAIL: NOT sorted: lst[%d]=%x lst[%d]=%x =====\n",
            i,lst[i],i+1,lst[i+1]);
  }
  return flag;
}


__global__ void scan_1_pass_cuda_func(int, int *dst, int *src, int n, int,
                                     int *blk_sums_prefix, int *blk_sums, int *flags, int) {
  __shared__ int bid;
  __shared__ int prefix[THREADS];
  __shared__ int block_offset;
  int tid = threadIdx.x;

  // Each block can wait for a predecessor that has already started.
  if (tid == 0) bid = atomicAdd(&d_blk_ctr, 1);
  __syncthreads();

  int i = bid * blockDim.x + tid;
  prefix[tid] = (i < n) ? src[i] : 0;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset *= 2) {
    int add = (tid >= offset) ? prefix[tid - offset] : 0;
    __syncthreads();
    prefix[tid] += add;
    __syncthreads();
  }

  if (tid == 0) {
    blk_sums[bid] = prefix[blockDim.x - 1];
    block_offset = 0;
    if (bid > 0) {
      while (atomicAdd(&flags[bid - 1], 0) == 0) {}
      __threadfence();
      block_offset = atomicAdd(&blk_sums_prefix[bid - 1], 0);
    }
    atomicExch(&blk_sums_prefix[bid], block_offset + blk_sums[bid]);
    __threadfence();
    atomicExch(&flags[bid], 1);
  }
  __syncthreads();

  if (i < n) dst[i] = block_offset + prefix[tid];
}

int radix_sort_cuda(int n, int *h_lst, int *, int *, int radix) {
  unsigned int bytes = sizeof(int) * n;
  int nthds = THREADS;
  int nblks = (n + nthds - 1) / nthds;
  unsigned int nblks_bytes = sizeof(int) * nblks;

  int *d_lst = nullptr, *d_buf = nullptr, *d_cnt = nullptr, *d_map = nullptr;
  int *blk_sums = nullptr, *blk_sums_prefix = nullptr, *d_flags = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_lst), bytes);
  cudaMalloc(reinterpret_cast<void **>(&d_buf), bytes);
  cudaMalloc(reinterpret_cast<void **>(&d_cnt), bytes);
  cudaMalloc(reinterpret_cast<void **>(&d_map), bytes);
  cudaMalloc(reinterpret_cast<void **>(&blk_sums), nblks_bytes);
  cudaMalloc(reinterpret_cast<void **>(&blk_sums_prefix), nblks_bytes);
  cudaMalloc(reinterpret_cast<void **>(&d_flags), nblks_bytes);

  cudaMemcpy(d_lst, h_lst, bytes, cudaMemcpyHostToDevice);

  /*
    this loop is the simplest it can get although it's inefficient
    each iteration sorts the lst using 1 bit
    think hard if you can sort using multiple bits
  */
  for (int iter=0;iter<MAXBIT;iter++) {
    cudaMemset(d_flags, 0, nblks_bytes);
    cudaMemcpyToSymbol(d_blk_ctr, &h_blk_ctr, sizeof(h_blk_ctr));

    cnt_cuda_func<<<nblks,nthds>>>(0,iter,n,d_buf,d_lst,d_cnt,d_map,radix);

    scan_1_pass_cuda_func<<<nblks,nthds>>>(0,d_map,d_cnt,n,nblks,blk_sums_prefix,blk_sums,d_flags,iter);

    int ones_total = 0;
    cudaMemcpy(&ones_total, d_map + n - 1, sizeof(ones_total), cudaMemcpyDeviceToHost);

    mov_cuda_func<<<nblks,nthds>>>(0,iter,n,d_buf,d_lst,d_cnt,d_map,radix,ones_total);

    int *previous = d_lst;
    d_lst = d_buf;
    d_buf = previous;
  }

  cudaMemcpy(h_lst, d_lst, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_lst);
  cudaFree(d_buf);
  cudaFree(d_cnt);
  cudaFree(d_map);
  cudaFree(blk_sums);
  cudaFree(blk_sums_prefix);
  cudaFree(d_flags);
  return 0;
}

/* End of file: radix_sort_cuda.c */
