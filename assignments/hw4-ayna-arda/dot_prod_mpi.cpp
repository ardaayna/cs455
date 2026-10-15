/* 
   CS698 GPU Cluster Programming
   Homework 4 MPI+CUDA on dot product
   10/21/2025
   Andrew Sohn
*/

// MPI include
#include <mpi.h>
#include <cmath>
#include <math.h>

// System includes
#include <iostream>
#include <sys/time.h>

using std::cout;
using std::cerr;
using std::endl;

// User include
#include "dot_decl.h"

#define MASTER 0
#define ROOT 0
#define MAX_ORDER 20	
#define MAX_N 1<<MAX_ORDER
#define MAX_PROCS 128
#define THREADS_PER_BLOCK 1024

#define MAX_BLOCK 1024		   /* 1024 for this box */
#define MAX_THREADS_PER_BLOCK 1024 /* 1024 for this box */

int vec_A[MAX_N], vec_B[MAX_N];

// Host code
int main(int argc, char *argv[]) {
  // Dimensions of the dataset
  int blockSize = 256;
  int gridSize = 10000;
  int dataSizePerNode = gridSize * blockSize;
  int i, n=0, order=10, max_n=0;
  int vec_size = MAX_ORDER;
  int my_work,my_rank,nprocs;;

  int my_prod=0,lst_prods[MAX_PROCS],prod=0,prod_host=0,prod_dev=0;

  MPI_Comm world = MPI_COMM_WORLD;

  long mpi_start, mpi_end, mpi_elapsed;
  long host_start, host_end, host_elapsed;
  long dev_start, dev_end, dev_elapsed;
  struct timeval timecheck;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (argc>1) {			/* order */
    order = atoi(argv[1]);
    n = 1 << order;
    printf("rank=%d: dot_prod: order=%d nprocs=%d\n",my_rank,order,nprocs);
    if (order > MAX_ORDER) {
      printf("rank=%d: order=%d > MAX_ORDER=%d for nprocs=%d: order set to %d\n",\
	     my_rank,order,MAX_ORDER,nprocs,MAX_ORDER);
      order = MAX_ORDER;
      n = MAX_N;
    }
    vec_size = n;
    printf("dot_prod: order=%d n=%d nprocs=%d\n", order, n, nprocs);
  }

  int log2_size = my_log(nprocs);

  my_work = n / nprocs;

  printf("rank=%d: nprocs=%d n=%d my_work=%d/%d=%d\n",my_rank,nprocs,n,n,nprocs,my_work);

  init_vec(vec_A, vec_size);
  init_vec(vec_B, vec_size);

  gettimeofday(&timecheck, NULL);
  mpi_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

  //  MPI_Scatter(vec_A, ... );
  MPI_Scatter(vec_A, my_work,MPI_INT,vec_A,my_work,MPI_INT,ROOT,MPI_COMM_WORLD);
  //  MPI_Scatter(vec_B, ... );
  MPI_Scatter(vec_B, my_work,MPI_INT,vec_B,my_work,MPI_INT,ROOT,MPI_COMM_WORLD);
  my_prod = dot_product_cuda(my_rank,my_work,vec_A,vec_B);

  //  MPI_Gather(...)
  MPI_Gather(&my_prod,1,MPI_INT,lst_prods,1,MPI_INT,ROOT, MPI_COMM_WORLD);
  prod_dev = sum(nprocs,lst_prods);

  gettimeofday(&timecheck, NULL);
  mpi_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  mpi_elapsed = mpi_end - mpi_start;

  if (my_rank == 0) {
    gettimeofday(&timecheck, NULL);
    host_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;
    prod_host = dot_product_host(nprocs,n,vec_A,vec_B);

    gettimeofday(&timecheck, NULL);
    host_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;
    host_elapsed = host_end - host_start;
 }

  MPI_Finalize();

  if (my_rank==0) {
    if (prod_host == prod_dev) 
      printf("\nTest Host: PASS: host=%x == dev=%x\n\n",prod_host,prod_dev);
    else
      printf("\nTest Host: FAIL: host=%x != dev=%x\n\n",prod_host,prod_dev);

    printf("************************************************\n");
    printf("mpi time: rank=%d: %d procs: %ld msecs\n",
	   my_rank, nprocs, mpi_elapsed);
    printf("host time: rank=%d: %d procs: %ld msecs\n",
	   my_rank, nprocs, host_elapsed);
    printf("************************************************\n");
  }

  return 0;
}

int dot_product_host(int nprocs, int n, int *x, int *y){
  int i=0,j=0,iii=0; int prod=0,bprod=0;
  int my_work;

  my_work = n/nprocs;

  int nblks = my_work / THREADS_PER_BLOCK;
  int nthds = THREADS_PER_BLOCK;
  if (n<THREADS_PER_BLOCK) {
    nblks = 1;
    nthds = my_work;
  }
  for (i=0;i<n;i++) prod = prod + *x++ * *y++;

  return prod;
}

// Initialize an array with random data (between 0 and 15)
void init_vec(int *data, int dataSize) {
  srand(time(NULL));
  for (int i = 0; i < dataSize; i++) *data++ = rand() & 0xf;
}

int my_log(int val) {
  int i;
  int log_val=0;

  for (i=val;i>1;i=i>>1) log_val++;

  return log_val;
}

/*************************************************
  End of file
*************************************************/
