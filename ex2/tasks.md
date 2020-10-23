# Exercise 2

## Task 1: Basic CUDA

Write a program that initializes two CUDA vectors (double precision floating point entries) of length N : One consisting of the numbers *0, 1, 2, 3, . . . , N − 1* and the other consisting of the numbers *N − 1, N − 2, . . . , 0* .

Investigate the following:

1. Measure the time to allocate (cudaMalloc) and free (cudaFree) a CUDA array for different
sizes N . (1 Point)
2. Compare the following three options to initialize the vectors:
    - Initialize directly within a dedicated CUDA kernel
    - Copy the data via cudaMemcpy() from a host array (e.g. from a malloc’ed array or from std::vector<double>).
    - Copy each individual entry by calling cudaMemcpy for each entry.
Provide the time to complete the initialization for each option and compare the effective bandwidth (in megabytes per second) obtained in each case. (1 Point)
3. Write a CUDA kernel that sums the two vectors. Make sure that the kernel works for different
values of N. (1 Point)
4. Measure and plot the execution time of the kernel for different values of N (e.g. 100, 300, 1000, 3000, 10000, 30000, 100000). What do you observe for small values of N and large values of N , respectively? (1 Point)
5. Try different grid sizes and block sizes as kernel launch parameters. For simplicity, consider the values 16, 32, 64, 128, 256, 512, and 1024. Which values lead to a significant reduction in performance for large (N = 10 7 )) vectors )? (1 point)