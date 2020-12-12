#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "timer.hpp"

// DEFINES
#define EX "ex8"
#define CSV_NAME "ph_data_mykernel_cuda.csv"

#define COUT
#define NUM_TEST 5
#define N_MIN 10
#define N_MAX 10000000 //1e7

//
//----------- Helper functions
//
template <template <typename, typename> class Container,
          typename ValueType,
          typename Allocator = std::allocator<ValueType>>
double median(Container<ValueType, Allocator> data)
{
    size_t size = data.size();
    if (size == 0)
        return 0.;
    sort(data.begin(), data.end());
    size_t mid = size / 2;

    return size % 2 == 0 ? (data[mid] + data[mid - 1]) / 2 : data[mid];
};

template <typename T>
double median(T *array, size_t size)
{
    if (size == 0)
        return 0.;
    sort(array, array + size);
    size_t mid = size / 2;

    return size % 2 == 0 ? (array[mid] + array[mid - 1]) / 2 : array[mid];
};
//
// my kernel
//
#define STRINGIFY(ARG) ARG
#define CUCL_KERNEL __global__                              // __kernel
#define CUCL_GLOBMEM                                        // __global
#define CUCL_LOCMEM __shared__                              // __local
#define CUCL_GLOBALID0 blockDim.x *blockIdx.x + threadIdx.x // get_global_id(0)
#define CUCL_GLOBALSIZE0 gridDim.x *blockDim.x              // get_global_size(0)
#define CUCL_LOCALSIZE0 blockDim.x                          // get_local_size(0)
#define CUCL_LOCALID0 threadIdx.x                           // get_local_id(0)
#define ATOMIC_ADD_FUNC atomicAdd                           // my_atomic_add

#define LOCAL_SIZE 256
#define BLOCK_SIZE LOCAL_SIZE
#define GRID_SIZE 256
#define GLOBAL_SIZE (BLOCK_SIZE*LOCAL_SIZE)

// // atomicAdd for OpenCL
// #ifndef ulong
//     #define ulong unsigned long
// #endif
// void my_atomic_add(volatile CUCL_GLOBMEM double *p, double val) {
//   volatile CUCL_GLOBMEM ulong* address_as_ul = (volatile CUCL_GLOBMEM ulong *) p;
//   volatile ulong old = *address_as_ul, assumed;
//   ulong val_as_ul =  (ulong) val;
//   do  {
//     assumed = old;
//     old = atomic_add(address_as_ul, val_as_ul);
//   } while (assumed != old);
// };

CUCL_KERNEL void initKernel(CUCL_GLOBMEM double *x, const uint N, const double val)
{
    const uint stride = CUCL_GLOBALSIZE0;
    uint gid = CUCL_GLOBALID0;

    for (; gid < N; gid += stride)
        x[gid] = val;
};

CUCL_KERNEL void some_asymmetry_relation(uint N, CUCL_GLOBMEM const double *x, CUCL_GLOBMEM const double *y, CUCL_GLOBMEM double *result)
{
    const uint stride = CUCL_GLOBALSIZE0;
    uint gid = CUCL_GLOBALID0;
    uint tid = threadIdx.x;
    CUCL_LOCMEM double cache[LOCAL_SIZE];

    double val = 0.0;
    for (; gid < N; gid += stride)
        val = (x[gid] + y[gid]) * (x[gid] - y[gid]);
    cache[tid] = val;

    __syncthreads();
    for (size_t i = CUCL_LOCALSIZE0 / 2; i != 0; i /= 2)
    {
        __syncthreads();
        if (tid < i)
            cache[tid] += cache[tid + i];
    }

    if (tid == 0)
        atomicAdd(result, cache[0]);
};

//
//----------- functions for this program
//
double benchmark(size_t N, double x_init, double y_init, std::vector<double> &results)
{

    Timer timer;
    timer.reset();

    std::vector<double> x(N, x_init);
    std::vector<double> y(N, y_init);

    double *X;
    double *Y;
    cudaMalloc(&X, N * sizeof(double));
    cudaMemcpy(X, x.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&Y, N * sizeof(double));
    cudaMemcpy(Y, y.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    results[0] = timer.get();

    double dot = 0.0;
    double *DOT;
    cudaMalloc(&DOT, sizeof(double));
    cudaMemcpy(DOT, &dot, sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> tmp(NUM_TEST, 0.0);
    for (int iter = 0; iter < NUM_TEST; iter++)
    {
        timer.reset();
        some_asymmetry_relation<<<GRID_SIZE, BLOCK_SIZE>>>();
    cudaMemcpy(&dot, DOT, sizeof(double), cudaMemcpyDeviceToHost);
        tmp[iter] = timer.get();
    }
    results[1] = median(tmp);

    double true_dot = (x_init + y_init) * (x_init - y_init) * N;

#ifdef COUT
    std::cout << "(x+y, x-y) = " << dot << " ?= " << true_dot << std::endl;
    std::cout << "Computation took " << results[1] << "s" << std::endl;
#endif
    timer.reset();
    results[3] = dot;
    results[2] = timer.get();

    cudaFree(X);
    cudaFree(Y);
    cudaFree(DOT);

    return dot;
}

int main(int argc, char const *argv[])
{
    double x_init = 1., y_init = 2.;
    std::vector<double> results(4, 0.0);

    std::ofstream csv;
    std::string sep = ";";
    std::string header = "N;vec_init_time;dot_time;memcpy_time;dot_result";
    auto to_csv = [&csv, &sep](double x) { csv << sep << x; };

    csv.open(CSV_NAME, std::fstream::out | std::fstream::trunc);
    csv << header << std::endl;
    for (size_t N = N_MIN; N < 1 + N_MAX; N *= 10)
    {
#ifdef COUT
        std::cout << "N: " << N << std::endl;
#endif
        benchmark(N, x_init, y_init, results);
        csv << N;
        std::for_each(results.begin(), results.end(), to_csv);
        csv << std::endl;
    }

    std::cout << "Data: https://gtx1080.360252.org/2020/" << EX << "/" << CSV_NAME;

    return EXIT_SUCCESS;
}
