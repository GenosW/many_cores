typedef double ScalarType;
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "timer.hpp"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Helper include file for error checking
#include "ocl-error.hpp"

// DEFINES
#define EX "ex8"
#define CSV_NAME "ph_data_mykernel_ocl.csv"

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
// #define STRINGIFY(ARG) ARG
#define CUCL_KERNEL __kernel
#define CUCL_GLOBMEM __global
#define CUCL_LOCMEM __local
#define CUCL_GLOBALID0 get_global_id(0)
#define CUCL_GLOBALSIZE0 get_global_size(0)
#define CUCL_LOCALSIZE0 get_local_size(0)
#define CUCL_LOCALID0 get_local_id(0)
#define CUCL_LOCBARRIER barrier(CLK_LOCAL_MEM_FENCE) //__syncthreads()
#define ATOMIC_ADD_FUNC my_atomic_add

// entry point, but need to account for multiple arguments AND need to actually force replacement before applying the macro
#define STRINGIFY(...) mFn2(__VA_ARGS__)
#define mFn2(ANS) #ANS

#define LOCAL_SIZE 128
#define BLOCK_SIZE LOCAL_SIZE
#define GRID_SIZE 128
#define GLOBAL_SIZE (GRID_SIZE * LOCAL_SIZE)

// atomicAdd for OpenCL
#ifndef ulong
    #define ulong unsigned long
#endif

std::string ocl_prog = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
"#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable\n"
STRINGIFY(void my_atomic_add(volatile CUCL_GLOBMEM double *p, double val) 
{
  volatile CUCL_GLOBMEM ulong* address_as_ul = (volatile CUCL_GLOBMEM ulong *) p;
  volatile ulong old = *address_as_ul; 
  volatile ulong assumed;
  ulong val_as_ul =  (ulong) val;
  do  {
    assumed = old;
    old = atomic_add(address_as_ul, val_as_ul);
  } while (assumed != old);
})""
"\n\n"
STRINGIFY(CUCL_KERNEL void initKernel(CUCL_GLOBMEM double *x, const uint N, const double val)
{
    const uint stride = CUCL_GLOBALSIZE0;
    uint gid = CUCL_GLOBALID0;

    for (; gid < N; gid += stride)
        x[gid] = val;
})
"\n\n"
STRINGIFY(CUCL_KERNEL void some_asymmetry_relation(uint N, CUCL_GLOBMEM const double *x, CUCL_GLOBMEM const double *y, CUCL_GLOBMEM double *result)
{
    const uint stride = CUCL_GLOBALSIZE0;
    uint gid = CUCL_GLOBALID0;
    uint lid = CUCL_LOCALID0;
    uint group_id = get_group_id(0);
    CUCL_LOCMEM double cache[LOCAL_SIZE];

    double val = 0.0;
    for (uint i = gid; i < N; i += stride)
        val += (x[i] + y[i]) * (x[i] - y[i]);
    cache[lid] = val;

    for (size_t i = CUCL_LOCALSIZE0 / 2; i != 0; i /= 2)
    {
        CUCL_LOCBARRIER;
        if (lid < i)
            cache[lid] += cache[lid + i];
    }

    if (lid == 0)
        result[group_id] = cache[lid]; // KERNEL_ARRAY
        // ATOMIC_ADD_FUNC(result, cache[0]); // KERNEL_ATOMIC
})
"\n\n";

// To test the atomic kernel version, simply switch the lines above 
// and comment out the define below
#define KERNEL_ARRAY
#ifndef KERNEL_ARRAY
    #define KERNEL_ATOMIC
#endif


//
//----------- functions for this program
//
double benchmark(
    cl_context& context, cl_command_queue& queue, cl_kernel& kernel, 
    size_t N, double x_init, double y_init, std::vector<double> &results)
{
    cl_int err;
    Timer timer;

    size_t local_size = LOCAL_SIZE;
    size_t global_size = GLOBAL_SIZE;
    size_t groups = 1 + int(N/LOCAL_SIZE);
    // std::cout << "LOCAL_SIZE: " << LOCAL_SIZE << std::endl;
    // std::cout << "GLOBAL_SIZE: " << GLOBAL_SIZE << std::endl;
    // std::cout << "groups: " << groups << std::endl;

    double dot = 0.0;
#ifdef KERNEL_ARRAY
    size_t dot_group_size = GRID_SIZE;
#endif

#ifdef KERNEL_ATOMIC
    size_t dot_group_size = 1;
#endif
    // std::vector<double> dot_group_results(N, dot);
    std::vector<double> dot_group_results(dot_group_size, dot);

    timer.reset();

    std::vector<double> x(N, x_init);
    std::vector<double> y(N, y_init);


    cl_mem X = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(double), x.data(), &err); 
    OPENCL_ERR_CHECK(err);
    cl_mem Y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(double), y.data(), &err); 
    OPENCL_ERR_CHECK(err);

    results[0] = timer.get();

// #ifdef KERNEL_ARRAY
    cl_mem DOT  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double)*dot_group_results.size(), (double*)dot_group_results.data(), &err); OPENCL_ERR_CHECK(err);
// #endif
    // DOT = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double), &dot, &err); OPENCL_ERR_CHECK(err);

    cl_uint vector_size = N;
// #ifdef KERNEL_ATOMIC
//     // cl_double DOT = dot;
//     cl_mem DOT = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double), &dot, &err); OPENCL_ERR_CHECK(err);
// #endif

    err = clSetKernelArg(kernel, 0, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem),  (double*)&X);OPENCL_ERR_CHECK(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem),  (double*)&Y); OPENCL_ERR_CHECK(err);
// #ifdef KERNEL_ATOMIC
//     err = clSetKernelArg(kernel, 3, sizeof(cl_mem),  (double*)&DOT); OPENCL_ERR_CHECK(err);
// #endif
// #ifdef KERNEL_ARRAY
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem),  (double*)&DOT); OPENCL_ERR_CHECK(err);
// #endif

    std::vector<double> tmp(NUM_TEST, 0.0);
    for (int iter = 0; iter < NUM_TEST; iter++)
    {
        dot = 0.0;
        timer.reset();
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

        err = clFinish(queue); 
        OPENCL_ERR_CHECK(err);
#ifdef KERNEL_ARRAY
        err = clEnqueueReadBuffer(queue, DOT, CL_TRUE, 0, dot_group_results.size()*sizeof(double), dot_group_results.data(), 0, NULL, NULL); 
        OPENCL_ERR_CHECK(err);

        // wait for all operations in queue to finish:
        err = clFinish(queue); 
        OPENCL_ERR_CHECK(err);

        for(auto& g: dot_group_results)
            dot += g;
#endif

        tmp[iter] = timer.get();
    }
    results[1] = median(tmp);

    double true_dot = (x_init + y_init) * (x_init - y_init) * N;

    timer.reset();
#ifdef KERNEL_ATOMIC
    err = clEnqueueReadBuffer(queue, DOT, CL_TRUE, 0, dot_group_results.size()*sizeof(double), dot_group_results.data(), 0, NULL, NULL); 
    OPENCL_ERR_CHECK(err);
    err = clFinish(queue); 
    OPENCL_ERR_CHECK(err);
    dot = dot_group_results[0];
#endif
    results[3] = dot;
    results[2] = timer.get();

#ifdef COUT
    std::cout << "(x+y, x-y) = " << dot << " ?= " << true_dot << std::endl;
    std::cout << "Computation took " << results[1] << "s" << std::endl;
#endif

    clReleaseMemObject(X);
    clReleaseMemObject(Y);
    clReleaseMemObject(DOT);

    return dot;
}

int main(int argc, char const *argv[])
{
    cl_int err;
    std::string target = "GPU";

    //
    /////////////////////////// Part 1: Set up an OpenCL context with one device ///////////////////////////////////
    //

    //
    // Query platform:
    //
    cl_uint num_platforms;
    cl_platform_id platform_ids[42];   //no more than 42 platforms supported...
    err = clGetPlatformIDs(42, platform_ids, &num_platforms); OPENCL_ERR_CHECK(err);
    std::cout << "# Platforms found: " << num_platforms << std::endl;


    //
    // Query devices:
    //
    cl_device_id device_ids[42];
    cl_uint num_devices;
    char device_name[64];
    cl_device_id my_device_id;
    cl_platform_id my_platform;
    for (int i = 0; i < num_platforms; ++i)
    {
        my_platform = platform_ids[i];
        if (target == "GPU") {
        err = clGetDeviceIDs(my_platform, CL_DEVICE_TYPE_GPU, 42, device_ids, &num_devices); 
        }
        else {
        err = clGetDeviceIDs(my_platform, CL_DEVICE_TYPE_CPU, 42, device_ids, &num_devices); 
        }
        if (err == CL_SUCCESS)
        break;
    } 
    OPENCL_ERR_CHECK(err);
    std::cout << "# Devices found: " << num_devices << std::endl;
    my_device_id = device_ids[0];

    size_t device_name_len = 0;
    err = clGetDeviceInfo(my_device_id, CL_DEVICE_NAME, sizeof(char)*63, device_name, &device_name_len); OPENCL_ERR_CHECK(err);

    std::cout << "Using the following device: " << device_name << std::endl;

    //
    // Create context:
    //
    cl_context my_context = clCreateContext(0, 1, &my_device_id, NULL, NULL, &err); OPENCL_ERR_CHECK(err);


    //
    // create a command queue for the device:
    //
    cl_command_queue my_queue = clCreateCommandQueueWithProperties(my_context, my_device_id, 0, &err); OPENCL_ERR_CHECK(err);


    //
    /////////////////////////// Part 2: Create a program and extract kernels ///////////////////////////////////
    //
    //
    // Build the program:
    //
    cl_kernel my_kernel;
    cl_program prog;

    const char * my_opencl_program = ocl_prog.c_str();

#ifdef COUT
    std::cout << "OpenCL program sources: " << std::endl << my_opencl_program << std::endl;
#endif
    size_t source_len = std::string(my_opencl_program).length();
    prog = clCreateProgramWithSource(my_context, 1, &my_opencl_program, &source_len, &err);OPENCL_ERR_CHECK(err);
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

    //
    // Print compiler errors if there was a problem:
    //
    if (err != CL_SUCCESS) {

        char *build_log;
        size_t ret_val_size;
        err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        build_log = (char *)malloc(sizeof(char) * (ret_val_size+1));
        err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = '\0'; // terminate string
        std::cout << "Log: " << build_log << std::endl;
        free(build_log);
        return EXIT_FAILURE;
    }

    my_kernel = clCreateKernel(prog, "some_asymmetry_relation", &err); OPENCL_ERR_CHECK(err);


    //
    // ------------ Benchmark setup ------------------
    // 
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
        benchmark(my_context, my_queue, my_kernel, N, x_init, y_init, results);
        csv << N;
        std::for_each(results.begin(), results.end(), to_csv);
        csv << std::endl;
    }

    std::cout << "Data: https://gtx1080.360252.org/2020/" << EX << "/" << CSV_NAME;


    clReleaseProgram(prog);
    clReleaseCommandQueue(my_queue);
    clReleaseContext(my_context);

    return EXIT_SUCCESS;
}
