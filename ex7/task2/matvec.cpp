#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "generate.hpp"
#include "timer.hpp"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Helper include file for error checking
#include "ocl-error.hpp"

// #ifndef uint
// // not defined for me...it's annoying
// using uint = uint32_t;
// #endif
typedef double ScalarType;

#define LOCAL_SIZE 128
#define GLOBAL_SIZE 128
#define NUM_TESTS 5
#define TARGET "GPU"
#define TRUNC_CSV
#define PP 16

#define SLOW_KERNEL
#ifndef SLOW_KERNEL
#define FAST_KERNEL
#endif
std::string target = TARGET;
#ifdef SLOW_KERNEL
std::string csv_name = "ph_data_" + target + ".csv";
#endif
#ifdef FAST_KERNEL
std::string csv_name = "ph_data2_" + target + ".csv";
#endif

std::string my_opencl_program =  "\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
" \n\n"
"__kernel void ocl_csr_matvec(uint N,\n"
"                        __global int *csr_rowoffsets,\n"
"                        __global int *csr_colindices,\n" 
"                        __global double *csr_values,\n"
"                        __global double const *x, __global double *y)\n"
"{\n"
"  uint gid = get_global_id(0);\n"
"  uint stride = get_global_size(0);\n"
"  for (size_t i=gid; i<N; i+=stride) {\n"
"    double value = 0;\n"
"    for (size_t j=csr_rowoffsets[i]; j<csr_rowoffsets[i+1]; ++j)\n"
"      value += csr_values[j] * x[csr_colindices[j]];\n"
"    y[i] = value;\n"
"  }\n"
"};"
" \n\n"
"__kernel void ocl_csr_matvec_fast(uint N,\n"
"                        __global int *csr_rowoffsets,\n"
"                        __global int *csr_colindices,\n" 
"                        __global double *csr_values,\n"
"                        __global double const *x, __global double *y)\n"
"{\n"
"  __local double cache[128];\n"
"  uint gid = get_group_id(0);\n"
"  uint lid = get_local_id(0);\n"
"  uint stride = get_num_groups(0);\n"
"  uint j_stride = get_global_size(0);\n"
"  for (size_t i=gid; i<N; i+=stride) {\n"
"    double value = 0;\n"
"    for (size_t j=csr_rowoffsets[i]; j<csr_rowoffsets[i+1]; j+=j_stride)\n"
"      value += csr_values[j] * x[csr_colindices[j]];\n"
"    cache[lid] = value;"
"    for (int i = get_local_size(0) / 2; i > 0; i /= 2) {\n"
"      barrier(CLK_LOCAL_MEM_FENCE);\n"
"      if (lid < i)\n"
"        cache[lid] += cache[lid + i];\n"
"    }\n"
"    if (lid==0)\n"
"      y[gid] = cache[0];\n"
"  }\n"
"};";

template <typename T>
void printContainer(T container, const int size) {
  std::cout << container[0];
  for (int i = 1; i < size; ++i) 
    std::cout << " | " << container[i] ;
  std::cout << std::endl;
}

template <typename T>
void printContainer(T container, const int size, const int only) {
  std::cout << container[0];
  for (int i = 1; i < only; ++i) 
      std::cout  << " | " << container[i];
  std::cout << " | ...";
  for (int i = size - only; i < size; ++i) 
    std::cout  << " | " << container[i];
  std::cout << std::endl;
}

double median(std::vector<double>& vec)
{
  size_t size = vec.size();
  if (size == 0)
          return 0.;
  sort(vec.begin(), vec.end());
  size_t mid = size/2;

  return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
};

bool check(const double* test, const double* ref, const uint N) {
  for (uint i = 0; i < N; ++i){
    if (test[i] != ref[i])
      return false;
  }
  return true;
}

double diff_norm(const double* test, const double* ref, const uint N) {
  double norm = 0.0;
  for (uint i = 0; i < N; ++i){
    norm += test[i] != ref[i];
  }
  return sqrt(norm);
}


/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y. CPU implementation.  */
void csr_matvec_product(size_t N,
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double const *x, double *y)
{
  for (size_t i=0; i<N; ++i) {
    double value = 0;
    for (size_t j=csr_rowoffsets[i]; j<csr_rowoffsets[i+1]; ++j)
      value += csr_values[j] * x[csr_colindices[j]];

    y[i] = value;
  }
}


double benchmark_ocl(size_t N, size_t max_nonzeros_per_row,
                  int* csr_rowoffsets, int *csr_colindices,
                  double* csr_values,
                  double* x, double* y)
{
  cl_int err;
  Timer timer1;
  //
  /////////////////////////// Part 1: Set up an OpenCL context with one device ///////////////////////////////////
  //

  //
  // Query platform:
  //
  cl_uint num_platforms;
  cl_platform_id platform_ids[42];   //no more than 42 platforms supported...
  err = clGetPlatformIDs(42, platform_ids, &num_platforms); OPENCL_ERR_CHECK(err);
  //std::cout << "# Platforms found: " << num_platforms << std::endl;

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
  //std::cout << "# Devices found: " << num_devices << std::endl;
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
  const char* ocl_prog = my_opencl_program.c_str();
  size_t source_len = my_opencl_program.length();

  cl_program prog = clCreateProgramWithSource(my_context, 1, &ocl_prog, &source_len, &err);OPENCL_ERR_CHECK(err);
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

#ifdef SLOW_KERNEL
  cl_kernel my_kernel = clCreateKernel(prog, "ocl_csr_matvec", &err); OPENCL_ERR_CHECK(err);
#endif
#ifdef FAST_KERNEL
  cl_kernel my_kernel = clCreateKernel(prog, "ocl_csr_matvec", &err); OPENCL_ERR_CHECK(err);
#endif
  
  //
  /////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
  //

  cl_uint vector_size = N;
  size_t  local_size = LOCAL_SIZE;
  size_t global_size = GLOBAL_SIZE*GLOBAL_SIZE;
  size_t groups = 1 + int(N/LOCAL_SIZE);

  cl_mem ocl_x = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(ScalarType)*N, x, &err); OPENCL_ERR_CHECK(err);

  cl_mem ocl_y = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(ScalarType)*N, y, &err); OPENCL_ERR_CHECK(err);

  cl_mem ocl_csr_rowoffsets = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*(N+1), csr_rowoffsets, &err); OPENCL_ERR_CHECK(err);

  cl_mem ocl_csr_colindices = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*(max_nonzeros_per_row*N), csr_colindices, &err); OPENCL_ERR_CHECK(err);

  cl_mem ocl_csr_values = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double)*(max_nonzeros_per_row*N), csr_values, &err); OPENCL_ERR_CHECK(err);

  //
  /////////////////////////// Part 4: Run kernel ///////////////////////////////////
  //
  //
  // Set kernel arguments:
  //
  
// "__kernel void ocl_csr_matvec(size_t N,\n"
// "                        __global int *csr_rowoffsets,\n"
// "                        __global int *csr_colindices,\n" 
// "                        __global double *csr_values,\n"
// "                        __global double const *x, __global double *y)\n"
  err = clSetKernelArg(my_kernel, 0, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem),  (double*)&ocl_csr_rowoffsets); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem),  (double*)&ocl_csr_colindices); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 3, sizeof(cl_mem),  (double*)&ocl_csr_values); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 4, sizeof(cl_mem),  (double*)&ocl_x); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 5, sizeof(cl_mem),  (double*)&ocl_y); OPENCL_ERR_CHECK(err);

  //
  // Enqueue kernel in command queue:
  //
  timer1.reset();
  err = clEnqueueNDRangeKernel(my_queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

  // wait for all operations in queue to finish:
  err = clFinish(my_queue); OPENCL_ERR_CHECK(err);
  double ocl_time = timer1.get();

  //
  /////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
  //
  err = clEnqueueReadBuffer(my_queue, ocl_y, CL_TRUE, 0, sizeof(ScalarType)*vector_size, y, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

  // wait for all operations in queue to finish:
  err = clFinish(my_queue); OPENCL_ERR_CHECK(err);
  //
  // cleanup
  //
  clReleaseMemObject(ocl_x);
  clReleaseMemObject(ocl_y);
  clReleaseMemObject(ocl_csr_rowoffsets);
  clReleaseMemObject(ocl_csr_colindices);
  clReleaseMemObject(ocl_csr_values);
  clReleaseProgram(prog);
  clReleaseCommandQueue(my_queue);
  clReleaseContext(my_context);

  std::cout << "From OCL benchmark: " << ocl_time << std::endl;

  return ocl_time;
}


/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void benchmark_matvec(size_t points_per_direction, size_t max_nonzeros_per_row,
                      void (*generate_matrix)(size_t, int*, int*, double*),
                      std::string gen_type) // function pointer parameter
{

  size_t N = points_per_direction * points_per_direction; // number of rows and columns
  std::fstream csv;
  csv.open(csv_name, std::fstream::out | std::fstream::app);

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets =    (int*)malloc(sizeof(double) * (N+1));
  int *csr_colindices =    (int*)malloc(sizeof(double) * max_nonzeros_per_row * N);
  double *csr_values  = (double*)malloc(sizeof(double) * max_nonzeros_per_row * N);

  //
  // fill CSR matrix with values
  //
  generate_matrix(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);

  //
  // Allocate vectors:
  //
  double *x = (double*)malloc(sizeof(double) * N); std::fill(x, x + N, 1);
  double *y = (double*)malloc(sizeof(double) * N); std::fill(y, y + N, 0);
  double *y_ocl = (double*)malloc(sizeof(double) * N); std::fill(y, y + N, 0);

  //
  // Call matrix-vector product reference
  //
  Timer timer;
  timer.reset();
  csr_matvec_product(N, csr_rowoffsets, csr_colindices, csr_values, x, y);
  double cpu_time = timer.get();

  //
  // Call matrix-vector product kernel
  //
  double ocl_time = benchmark_ocl(N, max_nonzeros_per_row, csr_rowoffsets, csr_colindices, csr_values, x, y_ocl);
  std::cout << "Reference: " << std::endl;
  printContainer(y, N, 10);
  std::cout << "OpenCL: " << std::endl;
  printContainer(y_ocl, N, 10);
  double difference = diff_norm(y_ocl, y, N);
  bool check_res = check(y_ocl, y, N);
  std::cout << "Difference between the two: " << difference << " (check: " << check_res << std::endl;
  std::cout << "Time for ref product: " << cpu_time << std::endl;
  std::cout << "Time for OCL product: " << ocl_time << std::endl;

  csv << N<< ";" 
      << points_per_direction << ";"
      << target << ";"
      << gen_type << ";"
      << ocl_time << ";" 
      << cpu_time << ";"
      << difference << ";"
      << check_res << std::endl;

  csv.close();
  free(x);
  free(y);
  free(y_ocl);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}


int main() {
  std::fstream csv;
#ifdef TRUNC_CSV
  csv.open(csv_name, std::fstream::out | std::fstream::trunc);
  csv << "N;points_per_direction;target;gen_type;ocl_time;cpu_time;check;diff_norm" << std::endl;
#endif
#ifndef TRUNC_CSV
  csv.open(csv_name, std::fstream::out | std::fstream::app);
#endif

  uint pp = PP;
  std::cout << "# Benchmarking finite difference matrix" << std::endl;
  benchmark_matvec(pp, 5, generate_fdm_laplace, "1"); // 100*100 unknowns, finite difference matrix
  // the last string is just so that I know which matrix was used

  std::cout << "# Benchmarking special matrix" << std::endl;
  benchmark_matvec(pp, 2000, generate_matrix2, "2");     // 100*100 unknowns, special matrix with 200-2000 nonzeros per row

  std::cout << "Data: \nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex7/" << csv_name << std::endl;

  return EXIT_SUCCESS;
}
