//
// Tutorial for demonstrating a simple OpenCL vector addition kernel
//
// Author: Karl Rupp    rupp@iue.tuwien.ac.at
//
typedef double ScalarType;
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Helper include file for error checking
#include "ocl-error.hpp"
#include "timer.hpp"

#define LOCAL_SIZE 128
#define GLOBAL_SIZE 128
#define NUM_TESTS 5

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
}

// const char *my_opencl_program =  "\n"
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
// " \n\n"
// "__kernel void xDOTy(__global double *result, \n"
// "                    __global double *x,\n"
// "                    __global double *y, \n"
// "                    __local cache, uint N) {\n"
// "  uint gid = get_global_id(0);\n"
// "  uint lid = get_local_id(0);\n"
// "  uint stride = get_global_size(0);\n"
// "  double tmp = 0.0;\n"
// "  for (uint i = gid; i < N; i += stride)\n"
// "    tmp = x[i] * y[i];\n"
// "  cache[lid] = tmp;\n"
// "  \n"
// "  for (int i = get_local_size(0) / 2; i > 0; i /= 2) {\n"
// "    barrier(CLK_LOCAL_MEM_FENCE);\n"
// "    if (lid < i)\n"
// "      cache[lid] += cache[lid + i];\n"
// "  }\n"
// "  if (lid==0)\n"
// "    result[get_group_id()] = cache[lid]\n"
// "};\n";


// "                    __local cache, uint N) {\n"

std::string my_opencl_program_c =  "\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
" \n\n"
"__kernel void xDOTy(__global double *result, \n"
"                    __global double *x,\n"
"                    __global double *y, \n"
"                    uint N) {\n"
"  uint gid = get_global_id(0);\n"
"  uint lid = get_local_id(0);\n"
"  uint stride = get_global_size(0);\n"
"  __local double cache[128];\n"
"  double tmp = 0.0;\n"
"  for (uint i = gid; i < N; i += stride)\n"
"    tmp += x[i] * y[i];\n"
"  cache[lid] = tmp;\n"
"  \n"
"  for (int i = get_local_size(0) / 2; i > 0; i /= 2) {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (lid < i)\n"
"      cache[lid] += cache[lid + i];\n"
"  }\n"
"  if (lid==0)\n"
"    result[get_group_id(0)] = cache[lid];\n"
"};\n";

std::string header = "\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
" \n\n";
std::string kernel_signature = "__kernel void xDOTy";
std::string kernel_code1 = "(__global double *result, \n"
"                    __global double *x,\n"
"                    __global double *y, \n"
"                    uint N) {\n";
std::string kernel_code2 = "  uint gid = get_global_id(0);\n"
"  uint lid = get_local_id(0);\n"
"  uint stride = get_global_size(0);\n"
"   __local double cache[128];\n"
"  double tmp = 0.0;\n"
"  for (uint i = gid; i < N; i += stride)\n"
"    tmp += x[i] * y[i];\n"
"  cache[lid] = tmp;\n"
"  \n"
"  for (int i = get_local_size(0) / 2; i > 0; i /= 2) {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (lid < i)\n"
"      cache[lid] += cache[lid + i];\n"
"  }\n"
"  if (lid==0)\n"
"    result[get_group_id(0)] = cache[lid];\n"
"};\n";


int main()
{
  cl_int err;

  bool compute = true;
  bool compile_M = false;
  std::vector<uint> M_vec{1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  if (!compile_M) {
    M_vec.clear();
    M_vec.push_back(1);
  }
  // std::string target = "GeForce GTX 1080";
  std::string target = "GPU";
  std::vector<uint> N_vec{256, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
  uint N_min = N_vec.front();
  uint N_max = N_vec.back();
  uint cnt = N_vec.size();


  Timer timer;
  timer.reset();

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


  std::fstream csv_compile;
  std::string csv_compile_name = "ph_data_compile.csv";

  if (compile_M) {
    csv_compile.open(csv_compile_name, std::fstream::out | std::fstream::trunc); 
    csv_compile << "M;compile_time;create_time" << std::endl;
  }
  int m = 1;

  std::string ocl_prog = header + kernel_signature + kernel_code1 + kernel_code2;
  for (auto& M : M_vec){

    // // To gernerate the M kernels
    for (; m < M; ++m) {
      ocl_prog += kernel_signature + std::to_string(m) + kernel_code1 + "uint insert" + std::to_string(m) + "=1;\n" + kernel_code2;
    }
    const char * my_opencl_program = ocl_prog.c_str();

    std::vector<double> tmp(NUM_TESTS, 0);
    for (uint iter = 0; iter < NUM_TESTS; iter++){
      timer.reset();
      size_t source_len = std::string(my_opencl_program).length();
      prog = clCreateProgramWithSource(my_context, 1, &my_opencl_program, &source_len, &err);OPENCL_ERR_CHECK(err);
      err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
      tmp[iter] = timer.get();
    }
    double compile_time = median(tmp);
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
    for (uint iter = 0; iter < NUM_TESTS; iter++){
      timer.reset();
      my_kernel = clCreateKernel(prog, "xDOTy", &err); OPENCL_ERR_CHECK(err);
      tmp[iter] = timer.get();
    }
    double create_time = median(tmp);
    std::cout << "Time to compile and create kernel: " << compile_time << std::endl;
    csv_compile << M << ";"
                << compile_time << ";"
                << create_time << std::endl;
  }


  // Plan to reuse all these vectors and buffers
  double y_val = 2., x_val=1.;
  std::vector<ScalarType> x(N_max, x_val);
  std::vector<ScalarType> y(N_max, y_val);
  std::vector<ScalarType> dot_vec(N_max, 0);

  cl_mem ocl_x = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N_max * sizeof(ScalarType), &(x[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_y = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N_max * sizeof(ScalarType), &(y[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_dot = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(ScalarType)*N_max, dot_vec.data(), &err); OPENCL_ERR_CHECK(err);

  //
  /////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
  //

  std::string csv_name = "ph_data_" + target + ".csv";
  std::fstream csv;
  if (compute){
    std::vector<double> times_cpu(N_vec.size(), 0.);
    std::vector<double> times_gpu(N_vec.size(), 0.);
    std::vector<double> times_total(N_vec.size(), 0.);
    csv.open(csv_name, std::fstream::out | std::fstream::trunc);
    csv << "N;target;local_size;global_size;ocl_time;cpu_time;total_time;dot" << std::endl;
    for (auto& N: N_vec){
      
      cl_uint vector_size = N;
      size_t  local_size = LOCAL_SIZE;
      size_t global_size = GLOBAL_SIZE*GLOBAL_SIZE;
      size_t groups = 1 + int(N/LOCAL_SIZE);

      //
      /////////////////////////// Part 4: Run kernel ///////////////////////////////////
      //
      

      //
      // Set kernel arguments:
      //
      // xDOTy(__global double *result, 
      //                   __global double *x,
      //                   __global double *y, 
      //                   __local cache, uint N)
      err = clSetKernelArg(my_kernel, 0, sizeof(cl_mem),  (double*)&ocl_dot); OPENCL_ERR_CHECK(err);
      err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem),  (double*)&ocl_x); OPENCL_ERR_CHECK(err);
      err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem),  (double*)&ocl_y); OPENCL_ERR_CHECK(err);
      // err = clSetKernelArg(my_kernel, 3, sizeof(cl_float) * local_work_size[0], NULL); OPENCL_ERR_CHECK(err);
      err = clSetKernelArg(my_kernel, 3, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);

      //
      // Enqueue kernel in command queue:
      //
      std::vector<double> tmp(NUM_TESTS, 0);
      for (uint iter = 0; iter < NUM_TESTS; iter++){
        timer.reset();
        err = clEnqueueNDRangeKernel(my_queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

        // wait for all operations in queue to finish:
        err = clFinish(my_queue); OPENCL_ERR_CHECK(err);
        tmp[iter] = timer.get();
      }
      double ocl_time = median(tmp);

      //
      /////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
      //

      // err = clEnqueueReadBuffer(my_queue, ocl_x, CL_TRUE, 0, sizeof(ScalarType) * x.size(), &(x[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);
      err = clEnqueueReadBuffer(my_queue, ocl_dot, CL_TRUE, 0, sizeof(ScalarType)*vector_size, dot_vec.data(), 0, NULL, NULL); OPENCL_ERR_CHECK(err);

      timer.reset();
      double dot = std::accumulate(dot_vec.begin(), dot_vec.begin()+groups, 0.);
      double cpu_time = timer.get();
      double total_time = ocl_time + cpu_time;


      std::cout << std::endl;
      std::cout << "Result of kernel execution: " << dot;
      std::cout << " =? " << vector_size*y_val*x_val << " : " << (dot == y_val*x_val*vector_size) << std::endl;
      std::cout << "Runtime: kernel + cpu = total_time" << std::endl;
      std::cout << ocl_time << " + " << cpu_time << " = " << total_time << std::endl;

      csv << N<< ";" 
          << target << ";"
          << local_size<< ";" 
          << global_size<< ";" 
          << ocl_time<< ";" 
          << cpu_time<< ";" 
          << total_time << ";"
          << dot << std::endl;
  

      // std::cout << "Result container:" << std::endl;
      // printContainer(dot_vec, dot_vec.size(), 10);
    }
    csv.close();
  }
  if (compile_M) csv_compile.close();
  //
  // cleanup
  //
  clReleaseMemObject(ocl_x);
  clReleaseMemObject(ocl_y);
  clReleaseMemObject(ocl_dot);
  clReleaseProgram(prog);
  clReleaseCommandQueue(my_queue);
  clReleaseContext(my_context);

  std::cout << "Data: \nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex7/" << csv_name << std::endl;
  std::cout << "Data: \nCompile times in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex7/" << csv_compile_name << std::endl;

  std::cout << std::endl;
  std::cout << "#" << std::endl;
  std::cout << "# My first OpenCL application finished successfully!" << std::endl;
  std::cout << "#" << std::endl;

  std::cout << "And here it is:" << std::endl;
  std::cout << ocl_prog;
  return EXIT_SUCCESS;
}

