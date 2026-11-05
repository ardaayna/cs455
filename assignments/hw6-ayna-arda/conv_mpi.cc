/*
  MPI+CUDA
  CS455 GPU Cluster Programming
  HW6 Convolution
  11/1/2025
  Andrew Sohn
*/

#include <mpi.h>
#include <cmath>
#include <math.h>

#include <iostream>
#include <sys/time.h>

using std::cout;
using std::cerr;
using std::endl;

#define MASTER 0
#define ROOT 0
#define MAX_N 1<<12
#define MAX_ORDER 1<<MAX_N
#define MAX_SIZE 1<<25
#define MAX_PROCS 64
#define THREADS_PER_BLOCK 256

#define TILE_WIDTH 4
#define MAX_TILE_WIDTH 16
#define MAX_BLOCK 1024		   /* 1024 for this box */
#define MAX_THREADS_PER_BLOCK 1024 /* 1024 for this box */
#define DEFAULT_DIM 256
#define DEFAULT_TILE 8
#define MASK_DIM 3
#define FILTER_DIM 3
#define FILTER_RADIUS 1
#define MAX_MASK_DIM 5


extern "C" {
  int conv_dev(int nprocs, int my_rank, int my_work, int *in_image,int *out_image, int height, int width, int filter_dim,int *filter_cpu);
}

// 32MB -> 8388608 (8M) ints
int in_image[1<<MAX_SIZE], out_image[1<<MAX_SIZE], out_image_host[1<<MAX_SIZE];

int filter_host[FILTER_DIM*FILTER_DIM];

void init_image(int flag, int *buf, int n) {
  srand(time(NULL));
  //  int cnt=0;
  //  for (int i = 0; i < dataSize; i++) *data++ = cnt++;
  if (flag==0) 
    for (int i = 0; i < n; i++) *buf++ = 0;
  else if (flag==1) 
    for (int i = 0; i < n; i++) *buf++ = 1;
  else
    for (int i = 0; i < n; i++) *buf++ = rand() & 0xf;
}

void conv_host(int* input, int* output, unsigned int height, unsigned int width) {
  int out_row,out_col,sum = 0;
  int filter_row,filter_col,in_row,in_col;

  for(out_row=0; out_row<height; out_row++) {
    for(out_col=0; out_col<width; out_col++) {
      sum = 0;
      /* Fill in */
      

      output[out_row*width + out_col] = sum;
    }
  }
}

int compare(int *dev, int *host, int height, int width) {
  int i,flag=1, row, col, n;
  n = height * width;

  for (i=0; i<n; i++) {
    if (*dev++ != *host++) {
      row = i/height;
      col = i%height;
      printf("DIFFERENT: dev[%d][%d]=%d != host[%d][%d]=%d\n",\
	     row,col,dev[i],row,col,host[i]);
      flag = 0;
      break;
    }
  }
  return flag;
}

void init_filter_host(int *buf) {
  int i,j;
  for (i=0; i<FILTER_DIM; i++) 
    for (j=0; j<FILTER_DIM; j++) *buf++ = 1;
}

int main(int argc, char *argv[]) {
  int i, n=0, order=0, max_n=0;
  int image_size = MAX_ORDER;
  int my_work,my_rank,nprocs;;
  int in_size=0,out_size=0,out_host_size=0;
  int height,width,filter_dim;

  MPI_Comm world = MPI_COMM_WORLD;

  long mpi_start, mpi_end, mpi_elapsed;
  long host_start, host_end, host_elapsed;
  long dev_start, dev_end, dev_elapsed;
  struct timeval timecheck;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int gx_dim, gy_dim, bx_dim, by_dim, tile_width;
  if (argc==4) {
    i=1;

    width = atoi(argv[i++]);
    height = atoi(argv[i++]);
    filter_dim = atoi(argv[i++]);

    if (width*height > 1 << MAX_SIZE) {
      printf("ERROR: width=%d * height=%d > 1 << MAX_SIZE=%d = %d\n",\
	     width,height,MAX_SIZE,1 << MAX_SIZE);
      return 1;
    }
    if (!((filter_dim==3 ) || (filter_dim=5))) {
      filter_dim = FILTER_DIM;		/* 3 */
      printf("INFO: filter_dim is set to DEFAULT_FILTER_DIM=%d\n",filter_dim);
    }
  }
  printf("rank=%d: image: width=%d x height=%d, filter_dim: %d x %d)\n",width,height,filter_dim,filter_dim);

  in_size = width * height;
  out_size = in_size;
  out_host_size = out_size;

  my_work = height / nprocs;

  printf("rank=%d: nprocs=%d width=%d my_work=%d/%d=%d\n",my_rank,nprocs,width,height,nprocs,my_work);

  init_image(1,in_image, in_size);   /* with random ints: 0..15 */
  init_image(0,out_image, out_size); /* zeros */
  init_filter_host(filter_host);

  gettimeofday(&timecheck, NULL);
  mpi_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

  /* MPI_Bcast in_image */
  /* ... */

  conv_dev(nprocs, my_rank, my_work, in_image,out_image, height, width, filter_dim,filter_host);

  /* MPI_Gather out_image */
  /* ... */

  gettimeofday(&timecheck, NULL);
  mpi_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  mpi_elapsed = mpi_end - mpi_start;

  if (my_rank == 0) {
    gettimeofday(&timecheck, NULL);
    host_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;
    conv_host(in_image,out_image_host, width, height);

    gettimeofday(&timecheck, NULL);
    host_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;
    host_elapsed = host_end - host_start;
 }

  MPI_Finalize();

  if (my_rank==0) {
    printf("************************************************\n");
    if (compare(out_image,out_image_host,height, width))
      printf("Test Host: PASS: host == dev\n");
    else
      printf("Test Host: FAIL: host == dev\n");
    
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
