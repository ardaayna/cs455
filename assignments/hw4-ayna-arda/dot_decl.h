#ifndef DOT_DECL_H
#define DOT_DECL_H

int sum(int size, int *data);
int my_log(int val);
void init_vec(int *data, int dataSize);

int dot_product_host(int nprocs, int n, int *x, int *y);
int dot_product_cpu(int rank, int n, int *x, int *y);
int dot_product_cuda(int my_rank, int my_work, int *h_A, int *h_B);

#endif