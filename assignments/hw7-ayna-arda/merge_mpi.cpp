/*
  MPI+CUDA Programming:
  Programming a cluster of CUDA-capable machines
  aka multi-node multi-GPU programming

  Andrew Sohn
  CS455/CS655 GPU Cluster Programming
  3/27/2025

  Three steps
  - Distribute data to procs
  - Local radix sort using CUDA
  - Merge locally independently sorted lists using MPI - no CUDA here

*/

#include <mpi.h>
#include <cmath>
#include <math.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <time.h>

using std::cout;
using std::cerr;
using std::endl;

#define MASTER 0
#define ROOT 0
#define MAX_N 1<<12
#define MAX_ORDER 25
#define MAX_SIZE 1 << MAX_ORDER
#define MAX_BIN 1 << 8		/* get two hexa's */

#define RADIX 4			/* 8 bits */

#define MAX_PROCS 64

#define ORG 1
#define LST 1
#define BUF 0
#define MAXBIT 32 

#define MAX_GRID_DIM 1<<12
#define MAX_BLOCK_DIM 1<<10
#define SH_GRID_DIM 1<<10	/* for shared mem declaration */
#define BLOCK_DIM 16

#define THREADS_PER_BLOCK 256

#define GRID_DIM 16
#define TILE_WIDTH 4
#define MAX_TILE_WIDTH 16
#define MAX_TILE MAX_TILE_WIDTH
#define MAX_BLOCK 1024		   /* 1024 for this box */
#define MAX_THREADS_PER_BLOCK 1024 /* 1024 for this box */
#define DEFAULT_DIM 256
#define DEFAULT_TILE 8

extern int sum(int rank, int size,int *data);
extern int radix_sort_cuda(int n, int *h_lst, int *h_cnt, int *h_map, int radix);

int lst[MAX_SIZE], buf[MAX_SIZE], tmp[MAX_SIZE], copy[MAX_SIZE],cnt[MAX_SIZE], map[MAX_SIZE];
int bin[1<<MAX_BIN],cpu_bin[1<<MAX_BIN],host_bin[1<<MAX_BIN];
int source,radix;   /* 1 for LST 0 for BUF holding the current data */

int correct(int source,int *org, int *buf, int n){
  int *slst,*dlst,zero,i,sidx;
  int flag=0;              /* 0 for no correction, 1 for correction */

  if (source == ORG) { slst = (int *) org; dlst = (int *) buf; }
  else { slst = (int *) buf; dlst = (int *) org; }

  for (i=0; i<n; i++) if (slst[i] < 0) break;

  zero=i;		       /* where numbers flipped: pos to neg */
  if (zero < n) {
    sidx=n-zero;
    for (i=0; i<zero; i++) dlst[sidx++] = slst[i];
    sidx=0;
    for (i=zero; i<n; i++) dlst[sidx++] = slst[i];
    flag=1;
  }
  return flag;
}

/* for radix=4,8,16,... */
void radix_sort_one_pass(int ii, int bucket, int n, const int *slst, int *dlst) {
  int i,j,mask;

  mask = bucket - 1;
  for (j=0;j<bucket;j++) cnt[j] = 0;

  for (j=0;j<n;j++) cnt[(static_cast<unsigned int>(slst[j]) >> ii) & mask]++;
  j = 0; 
  for (i=0;i<bucket;i++) {
    int count = cnt[i];
    cnt[i] = j;
    j += count;
  }
  for (i=0;i<n;i++) {
    int digit = (static_cast<unsigned int>(slst[i]) >> ii) & mask;
    dlst[cnt[digit]++] = slst[i];
  }
}

int radix_sort(int *lst, int n, int radix) {
  int i,flag=0;                 /* flag to see if corrected */
  int bucket = 1 << radix;

  source = LST;                 /* start with LST */
  for (i=0;i<MAXBIT;i=i+radix) {
    if (source == LST) radix_sort_one_pass(i,bucket,n,lst,buf);
    else radix_sort_one_pass(i,bucket,n,buf,lst);
    source = (~source) & 1;
  }

  flag = correct(source,lst,buf,n);
  if (flag) source = (~source) & 1;

  if (source == BUF) {
    for (i=0;i<n;i++) lst[i] = buf[i];
    source = (~source) & 1;
  }

  return flag;                  /* to indicate if corrected */
}


void print_lst(const char *name,int rank,int n, int *l){
  int i=0;
  printf("CPU rank=%d: %s: size=%d:: ",rank,name,n);
  for (i=0; i<n; i++) printf("%x ",l[i]);
  printf("\n");
}
/*Original code used logical and creating only 0 or 1, this uses bitwise and to avoid that. */
void init_lst(int *lst, int n, int mask) {
  srand(time(NULL));
  for (int i = 0; i < n; i++) *lst++ = rand() & mask;
}


int compare(int n, int *dev, int *host) {
  int i,flag=1, row, col,size;

  for (i=0; i<n; i++) {
    if (*dev++ != *host++) {
      printf("DIFFERENT: dev[%d]=%d != host[%d]=%d\n", i,dev[i],i,host[i]);
      flag = 0;
      break;
    }
  }
  return flag;
}

int my_log(int x) {
  int log=0;
  for (int i=x;i>1;i=i>>1) log++;
  return log;
}

void merge_sorted_lsts(int, int *left, int m, int *right, int n, int *out) {
  int i = 0, j = 0;
  for (int k = 0; k < m + n; ++k) {
    if (j == n || (i < m && left[i] <= right[j])) {
      out[k] = left[i++];
    } else {
      out[k] = right[j++];
    }
  }
}

/*
  Merge locally and independently sorted lists to a single list on MASTER
  Pair-wise send/recv
  iter0: stride=1  merge two immediate neighbors
  iter1: stride=2  merge two neighbors two hops away
  iter2: stride=4  merge two neighbors four hops away
  ...

  Changed for better handling
 */
int merge(int rank,int nprocs,int my_work, int *lst, MPI_Comm world) {
  int run_size = my_work;
  for (int stride = 1; stride < nprocs; stride *= 2) {
    if (rank % (2 * stride) != 0) {
      MPI_Send(lst, run_size, MPI_INT, rank - stride, 123, world);
      break;
    }

    int partner = rank + stride;
    if (partner < nprocs) {
      int right_size = std::min(stride, nprocs - partner) * my_work;
      MPI_Recv(buf, right_size, MPI_INT, partner, 123, world, MPI_STATUS_IGNORE);
      merge_sorted_lsts(rank, lst, run_size, buf, right_size, tmp);
      run_size += right_size;
      std::copy(tmp, tmp + run_size, lst);
    }
  }

  return 0;
}

static bool parse_int(const char *arg, int max, int &value) {
  char *end;
  long number = std::strtol(arg, &end, 10);
  if (end == arg || *end != '\0' || number < 1 || number > max) return false;
  value = static_cast<int>(number);
  return true;
}

static long time_ms() {
  struct timeval timecheck;
  gettimeofday(&timecheck, NULL);
  return (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
}

// No CUDA here, only MPI
/*Argument handling and actually checking results, using compare*/
int main(int argc, char *argv[]) {
  int n,my_work,rank,nprocs;
  int bits,mask;

  MPI_Comm world = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(world, &nprocs);
  MPI_Comm_rank(world, &rank);

  if (argc != 3 || !parse_int(argv[1], MAX_SIZE, n) ||
      !parse_int(argv[2], MAXBIT, bits) || nprocs > MAX_PROCS || n % nprocs != 0) {
    if (rank == MASTER) {
      fprintf(stderr, "Usage: %s <n: 1..%d, divisible by ranks> <bits: 1..%d>\n"
                      "Use at most %d MPI ranks.\n", argv[0], MAX_SIZE, MAXBIT, MAX_PROCS);
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  mask = (bits == MAXBIT) ? -1 : static_cast<int>((1U << bits) - 1);
  my_work = n / nprocs;
  radix = RADIX;

  printf("rank=%d: nprocs=%d n=%d my_work=%d bits=%d\n", rank,nprocs,n,my_work,bits);

  if (rank == MASTER) init_lst(copy,n,mask);
  MPI_Barrier(world);
  long mpi_start = time_ms();

  MPI_Scatter(copy,my_work,MPI_INT,lst,my_work,MPI_INT,MASTER,world);
  if (rank != MASTER) std::copy(lst, lst + my_work, copy);

  long dev_start = time_ms();
  radix_sort_cuda(my_work,lst,cnt,map,1);
  long dev_elapsed = time_ms() - dev_start;
  std::copy(lst, lst + my_work, map);

  merge(rank,nprocs,my_work,lst,world);
  long mpi_elapsed = time_ms() - mpi_start;

  std::copy(copy, copy + my_work, tmp);
  long cpu_start = time_ms();
  radix_sort(tmp,my_work,radix);
  long cpu_elapsed = time_ms() - cpu_start;
  int local_ok = compare(my_work,map,tmp);
  printf("rank=%d: CUDA vs CPU: %s (CUDA %ld msecs, CPU %ld msecs)\n",
         rank, local_ok ? "PASS" : "FAIL", dev_elapsed, cpu_elapsed);
  /*Add checks to actually print*/
  if (rank == MASTER) {
    std::copy(copy, copy + n, tmp);
    long host_start = time_ms();
    radix_sort(tmp,n,radix);
    long host_elapsed = time_ms() - host_start;
    std::sort(copy, copy + n);
    int host_ok = compare(n,tmp,copy);
    int merge_ok = compare(n,lst,tmp);
    printf("CPU radix vs std::sort: %s\n", host_ok ? "PASS" : "FAIL");
    printf("MPI+CUDA vs host: %s\n", merge_ok ? "PASS" : "FAIL");
    printf("MPI+CUDA time: %ld msecs; host radix time: %ld msecs\n", mpi_elapsed, host_elapsed);
    local_ok = local_ok && host_ok && merge_ok;
  }

  int all_ok;
  MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, world);

  MPI_Finalize();

  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*************************************************
  End of file: merge_mpi.cpp
*************************************************/
