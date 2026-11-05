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

#include <mpi.h>
#include <cmath>
#include <math.h>

#include <iostream>
#include <sys/time.h>

using std::cout;
using std::cerr;
using std::endl;

extern "C" {
  int matrix_multiply_cuda(int nprocs, int my_rank,int n, int my_work,int *h_A,int *h_B,int *h_C,int gx_dim,int gy_dim,int bx_dim,int by_dim );
}

#define MASTER 0
#define ROOT 0
#define MIN_ORDER 6		/* dim=256 */
#define MAX_ORDER 10		/* dim=4096 */
#define MIN_N 1<<MIN_ORDER
#define MAX_N 1<<MAX_ORDER
#define MAX_PROCS 32

#define MIN_TILE_WIDTH 4
#define MAX_TILE_WIDTH 16
#define MIN_BLOCK 32
#define MAX_BLOCK 512		/* 1024 for this box */

#define MIN_THREADS_PER_BLOCK 32
#define MAX_THREADS_PER_BLOCK 512 /* 1024 for this box */

#define MAX_BUF_SIZE 1<<25	/* 32MB -> 8388608 (8M) ints */
int mat_A[MAX_BUF_SIZE], mat_B[MAX_BUF_SIZE], mat_C[MAX_BUF_SIZE];
int mat_C_host[MAX_BUF_SIZE];

void init_mat(int *buf, int n) {
  srand(time(NULL));
  for (int i = 0; i < n; i++) *buf++ = rand() & 0xf;
}

int matrix_multiply(int *a, int *b, int *c, int n, int my_work) {
  /* 
     int i, j, k, sum=0;

     Fill function

 */
    int i, j, k, sum;
    for(i = 0; i < my_work; i++){
        for(k = 0; k < n; k++){
            sum  = 0;
            for(j = 0; j < n; j++){
                sum += a[i*n + j] * b[j*n + k];
            }
            c[i*n + k] = sum;
        }
    }

  return 0;
}

int compare(int n, int *dev, int *host) {
  int i,flag=1, row, col;
  int n_sq = n * n;

  for (i=0; i<n_sq; i++) {
    if (*dev++ != *host++) {
      row = i/n;
      col = i%n;
      printf("DIFFERENT: dev[%d][%d]=%d != host[%d][%d]=%d\n",\
	     row,col,dev[i],row,col,host[i]);
      flag = 0;
      break;
    }
  }
  return flag;
}

void print_lst_host(int name,int rank,int n, int *l){
  int i=0;
  printf("CPU rank=%d: %d: ",rank,name);
  for (i=0; i<n; i++) printf("%x ",l[i]);
  printf("\n");
}

int main(int argc, char *argv[]) {
  int i, n=0, order=0, max_n=0, n_sq=0;
  int my_work,my_rank,nprocs;
  int my_prod=0,lst_prods[MAX_PROCS],prod=0,prod_host=0,prod_dev=0;

  MPI_Comm world = MPI_COMM_WORLD;

  long mpi_start, mpi_end, mpi_elapsed;
  long host_start, host_end, host_elapsed;
  long dev_start, dev_end, dev_elapsed;
  struct timeval timecheck;
  int gx_dim, gy_dim, bx_dim, by_dim, tile_width;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (argc==3) {
    i=1;
    order = atoi(argv[i++]);
    if (order > MAX_ORDER) {
      printf("order=%d > MAX_ORDER=%d: order set to %d\n",\
	     order,MAX_ORDER,MAX_ORDER);
      order = MAX_ORDER;
    }

    tile_width = bx_dim = atoi(argv[i++]);
    if (tile_width>MAX_TILE_WIDTH) {
      tile_width=MAX_TILE_WIDTH;
      printf("tile_width set to MAX_TILE_WIDTH=%d\n",tile_width);
    }
  }else{
    order = MIN_ORDER;
    tile_width = MIN_TILE_WIDTH;
  }

  n = 1 << order;
  bx_dim = tile_width;
  by_dim = bx_dim;
  my_work = n / nprocs;
  //round up, necessary for cases where row count or matrix width isn't divisible by tile width
  gx_dim = (n + bx_dim - 1) / bx_dim;
  gy_dim = (my_work + by_dim - 1) / by_dim;
  printf("rank=%d: order=%d n=%d: grid(%d,%d), block(%d,%d)\n",my_rank, order, n, gx_dim, gy_dim, bx_dim,by_dim);

 

  printf("rank=%d: nprocs=%d n=%d my_work=%d/%d=%d\n",my_rank,nprocs,n,n,nprocs,my_work);

  n_sq = n*n;
  init_mat(mat_A, n_sq);
  init_mat(mat_B, n_sq);

  gettimeofday(&timecheck, NULL);
  mpi_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

  /* MPI_Scatter mat_A */
  MPI_Scatter(mat_A, my_work*n, MPI_INT,my_rank == ROOT ? MPI_IN_PLACE : mat_A, my_work*n, MPI_INT, ROOT, world);
  /* MPI_Bcast mat_B */
  MPI_Bcast(mat_B, n_sq, MPI_INT, ROOT, world);

  matrix_multiply_cuda(nprocs, my_rank, n, my_work, mat_A, mat_B, mat_C, gx_dim, gy_dim, bx_dim,by_dim);

  /* MPI_Gather mat_C */
  MPI_Gather(my_rank == ROOT ? MPI_IN_PLACE : mat_C, my_work*n, MPI_INT, mat_C, my_work*n, MPI_INT, ROOT, world);

  gettimeofday(&timecheck, NULL);
  mpi_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  mpi_elapsed = mpi_end - mpi_start;

  if (my_rank == 0) {
    gettimeofday(&timecheck, NULL);
    host_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;
    matrix_multiply(mat_A,mat_B,mat_C_host, n, n);
    gettimeofday(&timecheck, NULL);
    host_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;
    host_elapsed = host_end - host_start;
 }

  MPI_Finalize();

  if (my_rank==0) {
    if (compare(n,mat_C,mat_C_host))
      printf("\nTest Host: PASS: host == dev\n\n");
    else
      printf("\nTest Host: FAIL: host == dev\n\n");

    printf("************************************************\n");
    printf("mpi time: rank=%d: %d procs: %ld msecs\n",
	   my_rank, nprocs, mpi_elapsed);
    printf("host time: rank=%d: %d procs: %ld msecs\n",
	   my_rank, nprocs, host_elapsed);
    printf("************************************************\n");
  }

  return 0;
}

/*************************************************
  End of file
*************************************************/
