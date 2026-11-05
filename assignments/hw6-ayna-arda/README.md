CS455 GPU Cluster Programming MPI+CUDA - Homework 6
Homework 6 on Convolution in MPI and Cuda

Using the template files conv_mpi.cpp and conv_cuda.cu and Makefile, write a convolution program in MPI and CUDA. The two files, conv_mpi.cpp and conv_cuda.cu are taken from the working version, except some functions are deleted for you to fill. The files compile and the results actually pass the test of comparing the host version and the cuda version because they are initialiazed to zeros. You need to fill in the missing lines and functions as indicated in the files.

If you are not sure, ask. Or come to the office hours. See the class web page for office hours.

The master randomly generates the nxn input image, where n is an integer multiple of nprocs and a power of 2 number. The master broadcasts the entire input image to nprocs processes. You can try varying dimension such as nxm image once you get this simple version to work. This broadcasting eliminates scattering of variable rows of the input image because of the filter radius. I encourage you to try scatterv again after you got this simple version working.

For each process, perform convolution for n/nprocs rows on the device as well as on the cpu. Use the original input image location when computing convolution becasue it will simplify computation. However, when storing results, each process stores results starting at the base address of the output image. This will make comparing and gathering by the root/master process simpler. If you are not sure, ask. I will also briefly explain in class.

For each process, compare the CUDA results with the CPU ones. The CUDA results must match the CPU results. If they don't, do it until you match the two.

The master gathers n/nprocs rows from nprocs processes.

The master performs convolution on the entire image.

Compare the MPI+CUDA version gathered by root/master with the host version.

They must match. Otherwise, do it until you match the two.

This homework does not require you to use tiles which you must have mastered in HW5 matrix multiply. I strongly encourage you to use tiles after you get this simple version working. 