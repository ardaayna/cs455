/*
 * Matrix Multiplication
 * CS455 Intro to GPU Cluster Programming - MPI + CUDA
 * Fall 2026
 *
 * HW3 - One-Sided Communication
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXDIM 1<<12       /* 4096 */
#define ROOT 0

int mat_mult(double *A, double *B, double *C, int n, int my_work);
void init_data(double *data, int data_size);
int check_result(double *C, double *D, int n);

int main(int argc, char *argv[])
{
    int i;
    int n = 32, n_sq, flag, my_work;
    int my_rank, num_procs = 1;
    int elms_to_comm;

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *D = NULL;

    double *local_A;
    double *local_C;

    double start_time, end_time, elapsed;

    MPI_Win win_A, win_B, win_C;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc > 1) {
        n = atoi(argv[1]);

        if (n > MAXDIM)
            n = MAXDIM;
    }

    n_sq = n * n;

    /*
     * Assume n is evenly divisible by num_procs.
     */
    my_work = n / num_procs;          // number of rows assigned to each process
    elms_to_comm = my_work * n;       // number of matrix elements assigned to each process


    /*
     * ROOT stores the complete A and C matrices.
     */
    if (my_rank == ROOT) {
        A = (double *)malloc(sizeof(double) * n_sq);
        C = (double *)malloc(sizeof(double) * n_sq);
    }


    /*
     * Every process has storage for B and D.
     */
    B = (double *)malloc(sizeof(double) * n_sq);
    D = (double *)malloc(sizeof(double) * n_sq);


    /*
     * Local arrays for each process.
     */
    local_A = (double *)malloc(sizeof(double) * elms_to_comm);
    local_C = (double *)malloc(sizeof(double) * elms_to_comm);


    /*
     * ROOT initializes matrices A and B.
     */
    if (my_rank == ROOT) {

        printf("pid=%d: num_procs=%d n=%d my_work=%d\n",
               my_rank, num_procs, n, my_work);

        init_data(A, n_sq);
        init_data(B, n_sq);
    }


    /* TODO 1: Use MPI_Win_create() to create windows for A, B, and C */
    MPI_Win_create(A,my_rank == ROOT ? n_sq * sizeof(double) : 0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&win_A);
    MPI_Win_create(B,n_sq*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&win_B);
    MPI_Win_create(C,my_rank == ROOT ? n_sq * sizeof(double) : 0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&win_C);


    start_time = MPI_Wtime();
    MPI_Win_fence(0,win_A);
    /* TODO 2: Use MPI_Get() to get each process's rows of A from ROOT */
    MPI_Get(local_A,elms_to_comm,MPI_DOUBLE,ROOT,my_rank*elms_to_comm,elms_to_comm,MPI_DOUBLE,win_A);

    MPI_Win_fence(0,win_A);

    /* TODO 3: Use MPI_Put() to distribute B from ROOT to all processes */
    MPI_Win_fence(0,win_B);
    if(my_rank==ROOT){
        for(i = 1 ; i < num_procs; i++){
            MPI_Put(B,n_sq,MPI_DOUBLE,i,0,n_sq,MPI_DOUBLE,win_B);
        }
    }
    MPI_Win_fence(0,win_B);



    /* Local computation - already provided */
    mat_mult(local_A, B, local_C, n, my_work);


    /* TODO 4: Use MPI_Put() to put local_C into C on ROOT */

    MPI_Win_fence(0,win_C);
    MPI_Put(local_C,elms_to_comm,MPI_DOUBLE,ROOT,my_rank*elms_to_comm,elms_to_comm,MPI_DOUBLE,win_C);
    MPI_Win_fence(0,win_C);
    /*
     * ROOT verifies the result.
     */
    if (my_rank == ROOT) {

        end_time = MPI_Wtime();
        elapsed = end_time - start_time;

        /*
         * Sequential matrix multiplication for comparison.
         */
        mat_mult(A, B, D, n, n);

        flag = check_result(C, D, n);

        if (flag) {
            printf("Test: FAILED\n");
        }
        else {
            printf("Test: PASSED\n");
            printf("Total time %d: %f seconds.\n",
                   my_rank, elapsed);
        }
    }


    /*
     * Free allocated memory.
     */
    if (my_rank == ROOT) {
        free(A);
        free(C);
    }

    free(B);
    free(D);
    free(local_A);
    free(local_C);


    /* TODO 5: Use MPI_Win_free() to free the three windows */


    MPI_Win_free(&win_A);
    MPI_Win_free(&win_B);
    MPI_Win_free(&win_C);


    MPI_Finalize();

    return 0;
}


/*
 * Matrix multiplication.
 *
 * Each process computes my_work rows of C.
 */
int mat_mult(double *a, double *b, double *c, int n, int my_work)
{
    int i, j, k;
    double sum;

    for (i = 0; i < my_work; i++) {

        for (j = 0; j < n; j++) {

            sum = 0;

            for (k = 0; k < n; k++) {

                sum = sum +
                      a[i * n + k] *
                      b[k * n + j];
            }

            c[i * n + j] = sum;
        }
    }

    return 0;
}


/*
 * Initialize an array with random data.
 */
void init_data(double *data, int data_size)
{
    int i;

    for (i = 0; i < data_size; i++) {
        data[i] = rand() & 0xf;
    }
}


/*
 * Compare matrices C and D.
 */
int check_result(double *C, double *D, int n)
{
    int i, j;

    for (i = 0; i < n; i++) {

        for (j = 0; j < n; j++) {

            if (C[i * n + j] != D[i * n + j]) {

                printf("ERROR: C[%d][%d]=%f != D[%d][%d]=%f\n",
                       i, j, C[i * n + j],
                       i, j, D[i * n + j]);

                return 1;
            }
        }
    }

    return 0;
}
