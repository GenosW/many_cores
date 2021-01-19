
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <fstream>
#include "poisson2d.hpp"
#include "timer.hpp"
#include <CL/sycl.hpp>

// DEFINES
#define EX "ex10"
#define CSV_NAME "ph_data_sycl.csv"
#define N_MAX_PRINT 32
#define PRINT_ONLY 10
#define MAX_ITERS 100

#define GRID_SIZE 8
#define BLOCK_SIZE 8

/**by_sycl
 * 
 * namespace for my sycl functions to dinstinguish them from normal C++ (host) functions.
 * 
 * Design is inspired by the HIP/CUDA version.
 */
namespace by_sycl{
  namespace sycl = cl::sycl; // want to have shorter code.

  const int grid_size = GRID_SIZE;
  const int block_size = BLOCK_SIZE;

  /** ret = A*x
   * A... sparse matrix given in CSR format (3 arrays) of dense size NxN
   * x... vector of size A
   */
  void csr_Ax(sycl::queue& q,
              size_t N,
              sycl::buffer<int>& buff_csr_rowoffsets,
              sycl::buffer<int>& buff_csr_colindices,
              sycl::buffer<double>& buff_csr_values,
              sycl::buffer<double>& buff_x,
              sycl::buffer<double>& buff_y)
  {
    // std::cout << "Hello from by_sycl::csr_Ax!" << std::endl;
    // std::vector<double> y(a.size());

    // assert(("x should be length N", x.size() == N));
    // assert(("csr_rowoffsets should be length N+1", csr_rowoffsets.size() == (N+1)));
    

    // auto global_size = grid_size * block_size;
    // auto local_size = block_size;

    // sycl::nd_range<1> work_items(global_size, local_size);
    sycl::range<1> work_items{N};
    // Submitting a lambda to the queue that carries out the work: 
    q.submit( [&](sycl::handler& cgh)
    {
      auto rowoffsets = buff_csr_rowoffsets.get_access<sycl::access::mode::read>(cgh);
      auto colidx = buff_csr_colindices.get_access<sycl::access::mode::read>(cgh);
      auto values = buff_csr_values.get_access<sycl::access::mode::read>(cgh);
      auto x = buff_x.get_access<sycl::access::mode::read>(cgh);
      auto y = buff_y.get_access<sycl::access::mode::write>(cgh);

      // The parallel section
      cgh.parallel_for<class csr_Ax>(work_items,
                                        [=] (sycl::id<1> row) 
      {
        // for (int idx = row;)
        double tmp = 0;
        for (int jj = rowoffsets[row]; jj < rowoffsets[row+1]; ++jj) 
          tmp += values[jj] * x[colidx[jj]];
        y[row] = tmp;
      }
      //
      );
    });
  }

  /** x += alpha * y
   * 
   * x...     vector of size N
   * y...     vector of size N
   * alpha... scalar
   */
  void inc_ay(sycl::queue& q,
              const size_t N,
              sycl::buffer<double>& buff_x,
              sycl::buffer<double>& buff_y,
              const double alpha)
  {
    // std::cout << "Hello from by_sycl::inc_ay!" << std::endl;

    auto global_size = grid_size * block_size;
    auto local_size = block_size;

    sycl::nd_range<1> work_items(global_size, local_size);
    // Submitting a lambda to the queue that carries out the work: 
    q.submit( [&](sycl::handler& cgh)
    {
      auto x = buff_x.get_access<sycl::access::mode::read_write>(cgh);
      auto y = buff_y.get_access<sycl::access::mode::read>(cgh);

      // The parallel section
      cgh.parallel_for<class inc_ay_work>(work_items,
                                    [=] (sycl::nd_item<1> item) 
      {
        size_t tid = item.get_global_linear_id();
        if (tid < N)
          x[tid] = x[tid] + alpha * y[tid];
      });
    });
  }

  /** x = y + alpha * x
   * 
   * x...     vector of size N
   * y...     vector of size N
   * alpha... scalar
   */
  void update(sycl::queue& q,
              const size_t N,
              sycl::buffer<double>& buff_x,
              sycl::buffer<double>& buff_y,
              const double alpha)
  {
    // std::cout << "Hello from by_sycl::update!" << std::endl;

    auto global_size = grid_size * block_size;
    auto local_size = block_size;

    sycl::nd_range<1> work_items(global_size, local_size);
    // Submitting a lambda to the queue that carries out the work: 
    q.submit( [&](sycl::handler& cgh)
    {
      auto x = buff_x.get_access<sycl::access::mode::read_write>(cgh);
      auto y = buff_y.get_access<sycl::access::mode::read>(cgh);

      // The parallel section
      cgh.parallel_for<class update_work>(work_items,
                                        [=] (sycl::nd_item<1> item) 
      {
        size_t tid = item.get_global_linear_id();
        if (tid < N)
          x[tid] = y[tid] + alpha * x[tid];
      });
    });
  }


  /** dot = (x, y)
   * 
   * x...     vector of size N
   * y...     vector of size N
   */
  double dot(sycl::queue& q,
              const size_t N,
              sycl::buffer<double>& buff_x,
              sycl::buffer<double>& buff_y)
  {
    // std::cout << "Hello from by_sycl::dot!" << std::endl;
    double dot_result = 0;

    auto global_size = grid_size * block_size;
    auto local_size = block_size;

    sycl::nd_range<1> work_items(global_size, local_size);
    std::vector<double> result_host(grid_size, 0);
    { // this scope is for the result buffer to go out of scope "==" write to host
      sycl::buffer<double> buff_result(result_host.data(), result_host.size());
      q.submit( [&](sycl::handler& cgh)
      {
        auto x = buff_x.get_access<sycl::access::mode::read_write>(cgh);
        auto y = buff_y.get_access<sycl::access::mode::read>(cgh);
        auto result = buff_result.get_access<sycl::access::mode::write>(cgh);

        sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>(local_size), cgh);

        // The parallel section
        cgh.parallel_for<class dot_work>(
          work_items,
          [=] (sycl::nd_item<1> item) 
        {
          size_t local_id = item.get_local_linear_id();
          size_t global_id = item.get_global_linear_id();

          local_mem[local_id] = 0;
          for (int idx = global_id; idx < N; idx += global_size)
          {
            local_mem[local_id] = x[idx] * y[idx];
          }
          item.barrier(sycl::access::fence_space::local_space); 
          for (int k = local_size / 2; k > 0; k /= 2) 
          {
            item.barrier(sycl::access::fence_space::local_space); 
            if (local_id < k) 
            {
              local_mem[local_id] += local_mem[local_id + k];
            }
          }

          if (local_id == 0) 
          {
            result[item.get_group_linear_id()] = local_mem[0];
            // item.barrier(sycl::access::fence_space::local_space); 
            // std::cout << "[WG#: " << item.get_group_linear_id() << "]Number of work groups:" << item.get_group_range(0)  << std::endl;
            // std::cout << "[WG#: " << item.get_group_linear_id() << "local_mem[0] = " << local_mem[0] << std::endl;
          }
        });
      });
    }
    dot_result = std::accumulate(result_host.begin(), result_host.end(), 0);
    return dot_result;
  }
}

//
//// FUNCTIONS
//

namespace sycl = cl::sycl; // want to have shorter code.

/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse matrix-vector product) are transferred to CUDA kernels.
 */
double conjugate_gradient(size_t N,  // number of unknows
                        std::vector<int>& csr_rowoffsets, 
                        std::vector<int>& csr_colindices, 
                        std::vector<double>& csr_values,
                        std::vector<double>& rhs,
                        std::vector<double>& solution,
                        int& iters)
                        //, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  Timer timer;
  sycl::device device = sycl::default_selector{}.select_device();
  sycl::queue queue(device);

  // Check for potential local memory size
  auto has_local_mem = device.is_host()
      || (device.get_info<sycl::info::device::local_mem_type>()
      != sycl::info::local_mem_type::none);
  auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
  if (!has_local_mem || local_mem_size < (by_sycl::block_size * sizeof(size_t)))
  {
      throw "Device doesn't have enough local memory!";
  }

  // clear solution vector (it may contain garbage values):
  std::fill(solution.begin(), solution.end(), 1.);

  // initialize work vectors:
  std::vector<double> Ap(N, 0);
  // line 2: initialize r and p:
  std::vector<double> p(rhs);
  std::vector<double> r(rhs);

  // Initialize buffers for matrix and solution
  sycl::buffer<int> rowoffsets_buff(csr_rowoffsets.data(), csr_rowoffsets.size());
  sycl::buffer<int> colidx_buff(csr_colindices.data(), csr_colindices.size());
  sycl::buffer<double> values_buff(csr_values.data(), csr_values.size());

  sycl::buffer<double> solution_buff(solution.data(), solution.size());

  sycl::buffer<double> r_buff(r.data(), r.size());
  sycl::buffer<double> Ap_buff(Ap.data(), Ap.size());
  sycl::buffer<double> p_buff(p.data(), p.size());

  iters = 0;

  double initial_residual_squared = by_sycl::dot(queue, N, r_buff, r_buff);
  double res_norm = 0;
  timer.reset();
  while (1) {
    // line 4: A*p:
    // csr_matvec_product(N, csr_rowoffsets, csr_colindices, csr_values, p, Ap);
    by_sycl::csr_Ax(queue, 
                    N, rowoffsets_buff, colidx_buff, values_buff,
                    p_buff, Ap_buff);

    // lines 5,6:
    // double res_norm = 0;
    // for (size_t i=0; i<N; ++i) res_norm += r[i] * r[i];
    res_norm = by_sycl::dot(queue, N, r_buff, r_buff);

    // queue.wait_and_throw(); // necessary for Ap?

    // double alpha = 0;
    // for (size_t i=0; i<N; ++i) alpha += p[i] * Ap[i];
    double alpha = by_sycl::dot(queue, N, p_buff, Ap_buff);
    alpha = res_norm / alpha;
    // std::cout << "alpha(" << iters << ") = " << alpha << std::endl;

    // queue.wait_and_throw();

    // line 7,8:
    // for (size_t i=0; i<N; ++i) {
    //   solution[i] += alpha *  p[i];
    //   r[i]        -= alpha * Ap[i];
    // }
    by_sycl::inc_ay(queue, N, solution_buff, p_buff, alpha);
    by_sycl::inc_ay(queue, N, r_buff, Ap_buff, -alpha);

    double beta = res_norm;

    // lines 9, 10:
    // res_norm = 0;
    // for (size_t i=0; i<N; ++i) res_norm += r[i] * r[i];
    res_norm = by_sycl::dot(queue, N, r_buff, r_buff);
    if (std::sqrt( res_norm / initial_residual_squared ) < 1e-7) 
      break;

    // line 11: compute beta
    beta = res_norm / beta;

    // line 12: update p
    // for (size_t i=0; i<N; ++i) p[i] = r[i] + beta * p[i];
    by_sycl::update(queue, N, p_buff, r_buff, beta);
                    
    queue.wait_and_throw();
    if (iters > 1000) break;  // solver didn't converge
    ++iters;

  }
  double runtime = timer.get();
  std::cout << "Time elapsed: " << runtime << " (" << runtime / iters << " per iteration)" << std::endl;
  std::cout << "Norm in CG: " << res_norm << std::endl;

  if (iters > MAX_ITERS)
    std::cout << "Conjugate Gradient did NOT converge within " << MAX_ITERS << " iterations"
              << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations."
              << std::endl;

  return runtime;
}



/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void solve_system(size_t points_per_direction) {

  size_t N = points_per_direction * points_per_direction; // number of unknows to solve for

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
  //
  // int *csr_rowoffsets =    (int*)malloc(sizeof(double) * (N+1));
  // int *csr_colindices =    (int*)malloc(sizeof(double) * 5 * N);
  // double *csr_values  = (double*)malloc(sizeof(double) * 5 * N);
  std::vector<int> csr_rowoffsets(N+1);
  std::vector<int> csr_colindices(5*N);
  std::vector<double> csr_values(5*N);

  //
  // fill CSR matrix with values
  //
  std::cout << "Generating FDM..." << std::endl;
  generate_fdm_laplace(points_per_direction, csr_rowoffsets.data(), csr_colindices.data(), csr_values.data());

  //
  // Allocate solution vector and right hand side:
  //
  // double *solution = (double*)malloc(sizeof(double) * N);
  // double *rhs      = (double*)malloc(sizeof(double) * N);
  std::cout << "Initializing vectors..." << std::endl;
  std::vector<double> solution(N, 0);
  std::vector<double> rhs(N, 1.);
  // std::fill(rhs, rhs + N, 1);

  //
  // Call Conjugate Gradient implementation
  //
  int iters = 0;
  std::cout << "Starting CG" << std::endl;
  double runtime = conjugate_gradient(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution, iters);

  //
  // Check for convergence:
  //
  std::cout << "Calculating residual" << std::endl;
  double residual_norm = relative_residual(N, csr_rowoffsets.data(), csr_colindices.data(), csr_values.data(), rhs.data(), solution.data());
  std::cout << "Relative residual norm: " << residual_norm
            << " (should be smaller than 1e-6)" << std::endl;

  // for (auto & x : solution)
  //   std::cout << x << std::endl;

  // not optimal (efficient), but minimally invasive --> easy to copy
  std::ofstream csv;
  csv.open(CSV_NAME, std::fstream::out | std::fstream::app);
  csv << points_per_direction << ";" 
    << N << ";"
    << runtime << ";"
    << residual_norm << ";"
    << iters << std::endl;
  csv.close();

  // There were a bunch of frees missing anyway --> replaced with std::vectors
}


int main() {
  std::ofstream csv;
  csv.open(CSV_NAME, std::fstream::out | std::fstream::trunc);
  csv << "p;N;runtime;residual;iterations" << std::endl;
  csv.close();

  std::vector<int> p_per_dir{ 10, 100, 500,1000, 1500};

  for (auto& p : p_per_dir)
  {
    std::cout << "--------------------------" << std::endl;
    solve_system(p); // solves a system with p*p unknowns
  }
  std::cout << "\nData: https://gtx1080.360252.org/2020/" << EX << "/" << CSV_NAME;

  return EXIT_SUCCESS;
}
