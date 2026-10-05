#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long N = 400000000;

    if (N % size != 0) {
        if (rank == 0)
            printf("N must be divisible by number of processes\n");

        MPI_Finalize();
        return 1;
    }

    long long local_n = N / size;

    double *local_array =
        malloc(local_n * sizeof(double));

    double *array = NULL;

    if (rank == 0) {
        array = malloc(N * sizeof(double));

        for (long long i = 0; i < N; i++)
            array[i] = 1.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Distribution
    if (rank == 0) {

        // Rank 0 keeps the first chunk for itself
        for (long long i = 0; i < local_n; i++)
            local_array[i] = array[i];

        // Send remaining chunks to workers
        for (int dest = 1; dest < size; dest++) {

            MPI_Send(
                array + dest * local_n,
                local_n,
                MPI_DOUBLE,
                dest,
                0,
                MPI_COMM_WORLD
            );
        }

    } else {

        MPI_Recv(
            local_array,
            local_n,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    
    double local_sum = 0.0;

    for (long long i = 0; i < local_n; i++)
        local_sum += local_array[i];

    printf("Rank %d local sum = %.0f\n",
           rank, local_sum);

   
    double total_sum = 0.0;

    if (rank == 0) {

        total_sum = local_sum;

        for (int source = 1; source < size; source++) {

            double received_sum;

            MPI_Recv(
                &received_sum,
                1,
                MPI_DOUBLE,
                source,
                1,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            total_sum += received_sum;
        }

    } else {

        MPI_Send(
            &local_sum,
            1,
            MPI_DOUBLE,
            0,
            1,
            MPI_COMM_WORLD
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("\nTotal sum = %.0f\n", total_sum);
        printf("Processes = %d\n", size);
        printf("Execution time = %.6f seconds\n", end - start);
    }

    free(local_array);

    if (rank == 0)
        free(array);

    MPI_Finalize();

    // wall clock time
    // speed improvement
    // different inputs

    return 0;
}
