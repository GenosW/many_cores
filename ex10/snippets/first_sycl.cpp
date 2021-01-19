#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>

#include <CL/sycl.hpp>

//
// Worker routine for adding two vectors
// Performance note: Returning a vector will lead to a spurious copy (temporary) of the result
//
std::vector<double> add(cl::sycl::queue& q,
                           const std::vector<double>& a,
                           const std::vector<double>& b)
{
  std::vector<double> c(a.size());

  assert(a.size() == b.size());
  cl::sycl::range<1> work_items{a.size()};

  {
    cl::sycl::buffer<double> buff_a(a.data(), a.size());
    cl::sycl::buffer<double> buff_b(b.data(), b.size());
    cl::sycl::buffer<double> buff_c(c.data(), c.size());

    // Submitting a lambda to the queue that carries out the work: 
    q.submit([&](cl::sycl::handler& cgh){
      auto access_a = buff_a.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_b = buff_b.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_c = buff_c.get_access<cl::sycl::access::mode::write>(cgh);

      // The parallel section
      cgh.parallel_for<class vector_add>(work_items,
                                         [=] (cl::sycl::id<1> tid) {
        access_c[tid] = access_a[tid] + access_b[tid];
      });
    });
  }
  return c;
}

/**by_sycl
 * 
 * namespace for my sycl functions to dinstinguish them from normal C++ (host) functions.
 */
namespace by_sycl{
  namespace sycl = cl::sycl;

  const size_t grid_size = 8;
  const size_t block_size = 8;

  /** x = alpha * y
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
    std::cout << "Hello from by_sycl::inc_ay!" << std::endl;
    sycl::nd_range<1> work_items(grid_size * block_size, block_size);
    // Submitting a lambda to the queue that carries out the work: 
    q.submit( [&](sycl::handler& cgh)
    {
      auto x = buff_x.get_access<sycl::access::mode::write>(cgh);
      auto y = buff_y.get_access<sycl::access::mode::read>(cgh);

      // The parallel section
      cgh.parallel_for<class inc_ay>(work_items,
                                        [=] (sycl::nd_item<1> item) 
      {
        size_t tid = item.get_global_linear_id();
        if (tid < N)
          x[tid] += alpha * y[tid];
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
    std::cout << "Hello from by_sycl::update!" << std::endl;
    sycl::nd_range<1> work_items(grid_size * block_size, block_size);
    // Submitting a lambda to the queue that carries out the work: 
    q.submit( [&](sycl::handler& cgh)
    {
      auto x = buff_x.get_access<sycl::access::mode::write>(cgh);
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
    std::cout << "Hello from by_sycl::dot!" << std::endl;
    double dot_result = 0;

    auto global_size = grid_size * block_size;
    auto local_size = block_size;

    sycl::nd_range<1> work_items(global_size, block_size);
    std::vector<double> result_host(grid_size, 0);
    { // this scope is for the result buffer to go out of scope "==" write to host
      sycl::buffer<double> buff_result(result_host.data(), result_host.size());
      q.submit( [&](sycl::handler& cgh)
      {
        auto x = buff_x.get_access<sycl::access::mode::write>(cgh);
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

  namespace old {

    /**result = (x, y)
     *
     */
    double x_dot_y(sycl::queue& q,
                const size_t N,
                sycl::buffer<double>& buff_x,
                sycl::buffer<double>& buff_y)
    {
      std::cout << "Hello from by_sycl::x_dot_y!" << std::endl;

      int n_workgroups = int( (N+block_size) / block_size );
      // auto global_size = grid_size * block_size;
      auto global_size = n_workgroups * block_size;
      auto local_size = block_size;

      std::vector<double> result_host(n_workgroups, 0);
      { // this scope is for the result buffer to go out of scope "==" write to host
        sycl::buffer<double> buff_result(result_host.data(), result_host.size());

        sycl::nd_range<1> work_items(global_size, local_size);
        auto event = q.submit( [&](sycl::handler& cgh)
        {
          sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>(local_size), cgh);

          auto x = buff_x.get_access<sycl::access::mode::write>(cgh);
          auto y = buff_y.get_access<sycl::access::mode::read>(cgh);
          auto result = buff_result.get_access<sycl::access::mode::write>(cgh);

          cgh.parallel_for<class x_dot_y_work>(
                  work_items,
                  [=] (sycl::nd_item<1> item) 
          {// lambda 
            size_t local_id = item.get_local_linear_id();
            size_t global_id = item.get_global_linear_id();

            local_mem[local_id] = 0;
            for (int idx = global_id; idx < N; idx += global_size)
            {
              local_mem[local_id] = x[idx] * y[idx];
            }
            item.barrier(sycl::access::fence_space::local_space); 
            // for (int k = local_size / 2; k > 0; k /= 2) 
            // {
            //   item.barrier(sycl::access::fence_space::local_space); 
            //   if (local_id < k) 
            //   {
            //     local_mem[local_id] += local_mem[local_id + k];
            //   }
            // }

            if (local_id == 0) 
            {
              result[item.get_group_linear_id()] = local_mem[0];
              item.barrier(sycl::access::fence_space::local_space); 
              std::cout << "[WG#: " << item.get_group_linear_id() << "]Number of work groups:" << item.get_group_range(0)  << std::endl;
              std::cout << "[WG#: " << item.get_group_linear_id() << "local_mem[0] = " << local_mem[0] << std::endl;
            }
          });
        });
        event.wait_and_throw(); // doesn't wait for whole queue but waits for this event to be done so we can sum up the vector
      }// this scope is for the result buffer to go out of scope "==" write to host
      return std::accumulate(result_host.begin(), result_host.end(), 0);
    }


    /**result = (x, y)
     *
     */
    double x_dot_y2(sycl::queue& q,
                const size_t N,
                sycl::buffer<double>& buff_x,
                sycl::buffer<double>& buff_y)
    {
      std::cout << "Hello from by_sycl::x_dot_y!" << std::endl;
      std::vector<double> result_host(grid_size, 0);
      { // this scope is for the result buffer to go out of scope "==" write to host
        sycl::buffer<double> buff_result(result_host.data(), result_host.size());

        sycl::nd_range<1> work_items(grid_size * block_size, block_size);
        auto event = q.submit( [&](sycl::handler& cgh)
        {
          sycl::accessor<size_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>(block_size), cgh);

          auto x = buff_x.get_access<sycl::access::mode::write>(cgh);
          auto y = buff_y.get_access<sycl::access::mode::read>(cgh);
          auto result = buff_result.get_access<sycl::access::mode::write>(cgh);

          cgh.parallel_for<class x_dot_y_work>(
                  sycl::nd_range<1>(grid_size * block_size, block_size),
                  [=] (sycl::nd_item<1> item) 
          {// lambda 
            size_t local_id = item.get_local_linear_id();
            size_t global_id = item.get_global_linear_id();

            local_mem[local_id] = x[global_id] * y[global_id];

            for (int k = block_size / 2; k > 0; k /= 2) 
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
            }
          });
        });
        event.wait_and_throw(); // doesn't wait for whole queue but waits for this event to be done so we can sum up the vector
      }// this scope is for the result buffer to go out of scope "==" write to host
      return std::accumulate(result_host.begin(), result_host.end(), 0);
    }

  }
}

int main()
{
  cl::sycl::queue q;
  size_t N = 8;
  std::vector<double> a(N);
  std::vector<double> b(N);
  double flip = 1;
  for (size_t idx = 0; idx < N; idx++)
  {
    a[idx] = idx;
    b[idx] = flip * idx;
    flip *= -1;
    std::cout << idx << ": ( " << a[idx] << " | " << b[idx] << " )" << std::endl;
  }


  auto result = add(q, a, b);

  double dot_true;
  double dot;
  std::vector<double> c(N, 0.);
  std::vector<double> d = a;
  {
    cl::sycl::buffer<double> buff_a(a.data(), a.size());
    cl::sycl::buffer<double> buff_b(b.data(), b.size());
    cl::sycl::buffer<double> buff_c(c.data(), c.size());
    cl::sycl::buffer<double> buff_d(d.data(), d.size());

    // should produce same effect
    by_sycl::inc_ay(q, N, buff_c, buff_a, 1.);
    by_sycl::inc_ay(q, N, buff_c, buff_b, 1.);
    by_sycl::update(q, N, buff_d, buff_b, 1.);

    // try dot product
    dot_true = std::inner_product(a.begin(), a.end(), b.begin(), 0);

    // dot = by_sycl::x_dot_y(q, N, buff_a, buff_b);
    // std::cout << "Result of dot product x_dot_y(a, b ): " << dot << " =? " << dot_true << std::endl;

    dot = by_sycl::dot(q, N, buff_a, buff_b);
    std::cout << "Result of dot product dot(a,b ): " << dot << " =? " << dot_true << std::endl;

    // dot = by_sycl::x_dot_y2(q, N, buff_a, buff_b);
    // std::cout << "Result of dot product2 x_dot_y2(a,b ): " << dot << " =? " << dot_true << std::endl;
  }
  std::cout << "Result (first_sycl): " << std::endl;
  for(const auto& x: result)
    std::cout << x << std::endl;

  
  std::cout << "Result (by_sycl::inc_ay): " << std::endl;
  for(const auto& x: c)
    std::cout << x << std::endl;
  

  std::cout << "Result (by_sycl::vecadd2): " << std::endl;
  for(const auto& x: d)
    std::cout << x << std::endl;

  
}
