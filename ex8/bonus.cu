//
// Tutorial for demonstrating a simple OpenCL vector addition kernel
//
// Author: Karl Rupp    rupp@iue.tuwien.ac.at
//

typedef double       ScalarType;


#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Helper include file for error checking
#include "ocl-error.hpp"
#include "timer.hpp"

//
// Transformation to OpenCL
//
#define CUCL_KERNEL __kernel
#define CUCL_GLOBALMEM __global
#define CUCL_LOCALMEM __local
#define CUCL_GLOBALID0 get_global_id(0)
#define CUCL_GLOBALSIZE0 get_global_size(0)
#define CUCL_GROUPID0 get_group_id(0)
#define CUCL_LOCALSIZE0 get_local_size(0)
#define CUCL_LOCALID0 get_local_id(0)
#define CUCL_BARRIER barrier(CLK_LOCAL_MEM_FENCE) //__syncthreads()

// entry point, but need to account for multiple arguments AND need to actually force replacement before applying the macro
#define STRINGIFY(...) mFn2(__VA_ARGS__)
#define mFn2(ANS) #ANS

// ##define STRINGIFY(ARG)   #ARG

const char *my_opencl_program = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
#include "dot.cucl"
;

// undefine STRINGIFY after use to avoid global havoc:
#undef STRINGIFY
#undef mFn2


int dot_opencl(unsigned int N)
{
  cl_int err;

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
  cl_platform_id my_platform = platform_ids[0];


  //
  // Query devices:
  //
  cl_device_id device_ids[42];
  cl_uint num_devices;
  err = clGetDeviceIDs(my_platform, CL_DEVICE_TYPE_ALL, 42, device_ids, &num_devices); OPENCL_ERR_CHECK(err);
  std::cout << "# Devices found: " << num_devices << std::endl;
  cl_device_id my_device_id = device_ids[0];

  char device_name[64];
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

  Timer timer;
  timer.reset();

  //
  // Build the program:
  //
  size_t source_len = std::string(my_opencl_program).length();
  cl_program prog = clCreateProgramWithSource(my_context, 1, &my_opencl_program, &source_len, &err);OPENCL_ERR_CHECK(err);
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
    std::cout << "OpenCL program sources: " << std::endl << my_opencl_program << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Extract the only kernel in the program:
  //
  cl_kernel my_kernel = clCreateKernel(prog, "dotProduct", &err); OPENCL_ERR_CHECK(err);

  std::cout << "Time to compile and create kernel: " << timer.get() << std::endl;


  //
  /////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
  //

  //
  // Set up buffers on host:
  //
  cl_uint vector_size = N;
  std::vector<ScalarType> x(vector_size, 1.0);
  std::vector<ScalarType> y(vector_size, 2.0);
  std::vector<ScalarType> partial(vector_size, 0.0);

  //
  // Now set up OpenCL buffers:
  //
  cl_mem ocl_x       = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(x[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_y       = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(y[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_partial = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256 * sizeof(ScalarType), &(y[0]), &err); OPENCL_ERR_CHECK(err);



  //
  /////////////////////////// Part 4: Run kernel ///////////////////////////////////
  //
  size_t  local_size = 256;
  size_t global_size = 256*256;

  //
  // Set kernel arguments:
  //
  err = clSetKernelArg(my_kernel, 0, sizeof(cl_mem),  (void*)&ocl_x); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem),  (void*)&ocl_y); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem),  (void*)&ocl_partial); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 3, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);

  //
  // Enqueue kernel in command queue:
  //
  err = clEnqueueNDRangeKernel(my_queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

  // wait for all operations in queue to finish:
  err = clFinish(my_queue); OPENCL_ERR_CHECK(err);


  //
  /////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
  //

  err = clEnqueueReadBuffer(my_queue, ocl_partial, CL_TRUE, 0, sizeof(ScalarType) * 256, &(partial[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);

  ScalarType dot = 0;
  for (size_t i=0; i<256; ++i) dot += partial[i];

  std::cout << "Result of OpenCL: " << dot << std::endl;

  //
  // cleanup
  //
  clReleaseMemObject(ocl_x);
  clReleaseMemObject(ocl_y);
  clReleaseMemObject(ocl_partial);
  clReleaseProgram(prog);
  clReleaseCommandQueue(my_queue);
  clReleaseContext(my_context);

  return 0;
}



//
// CUDA version
//
#define CUCL_KERNEL __global__                              // __kernel
#define CUCL_GLOBALMEM                                         // __global
#define CUCL_LOCALMEM __shared__                              // __local
#define CUCL_GLOBALID0 blockDim.x *blockIdx.x + threadIdx.x // get_global_id(0)
#define CUCL_GLOBALSIZE0 gridDim.x *blockDim.x              // get_global_size(0)
#define CUCL_GROUPID0 blockIdx.x
#define CUCL_LOCALSIZE0 blockDim.x                          // get_local_size(0)
#define CUCL_LOCALID0 threadIdx.x                           // get_local_id(0)
#define CUCL_BARRIER __syncthreads()

#define STRINGIFY(...) mFn2(__VA_ARGS__)
#define mFn2(ANS) ANS

#include "dot.cucl"
// STRINGIFY( CUCL_KERNEL void dotProduct(
//   CUCL_GLOBALMEM double *x,
//   CUCL_GLOBALMEM double *y,
//   CUCL_GLOBALMEM double * partial_result,
//   unsigned int N)
// {
//   CUCL_LOCALMEM double shared_buf[512]; double thread_sum = 0;
//   for (int i = CUCL_GLOBALID0; i < N; i += CUCL_GLOBALSIZE0)
//     thread_sum += x[i]* y[i];

//   shared_buf[CUCL_LOCALID0] = thread_sum;
//   for (int stride = CUCL_LOCALSIZE0 / 2; stride > 0; stride /= 2) {
//     CUCL_BARRIER;
//     if (CUCL_LOCALID0 < stride)
//       shared_buf[CUCL_LOCALID0] += shared_buf[CUCL_LOCALID0+stride];
//   }

//   CUCL_BARRIER;
//   if (CUCL_LOCALID0 == 0)
//     partial_result[CUCL_GROUPID0] = shared_buf[0];
// } )
// TODO for you: Define CUCL preprocessor directives to import proper kernel from dot.cucl

int dot_cuda(unsigned int N) {

  double *cuda_x, *cuda_y, *cuda_partial;
  cudaMalloc(&cuda_x, N * sizeof(double));
  cudaMalloc(&cuda_y, N * sizeof(double));
  cudaMalloc(&cuda_partial, 256*sizeof(double));

  // create x, y, and temporary vectors
  std::vector<double> x(N, 1.0), y(N, 2.0), partial(256, 0.0);
  cudaMemcpy(cuda_x, &x[0], N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_y, &y[0], N*sizeof(double), cudaMemcpyHostToDevice);

  dotProduct<<<256, 256>>>(cuda_x, cuda_y, cuda_partial, N);

  // get partial results back and sum on CPU
  cudaMemcpy(&partial[0], cuda_partial, 256*sizeof(double), cudaMemcpyDeviceToHost);
  double dot = 0;
  for (size_t i=0; i<256; ++i) dot += partial[i];

  std::cout << "Result of CUDA dot: " << dot << std::endl;

  return 0;
}


//
// Main execution flow:
//
int main() {

  unsigned int N = 1000;

  // runtime switches:
  int use_opencl = 1;
  int use_cuda = 1;
 
  if (use_opencl) {
    std::cout << "Running OpenCL version" << std::endl;
    dot_opencl(N);
  }

  if (use_cuda) {
    std::cout << std::endl << "Running CUDA version" << std::endl;
    dot_cuda(N);
  }

  return EXIT_SUCCESS;
}

