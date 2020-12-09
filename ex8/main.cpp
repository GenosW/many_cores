#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cstdlib>
#include "timer.hpp"

// DEFINES
#define EX "ex8"
#define CSV_NAME "ph_data.csv"
#define COUT
#define NUM_TEST 5
#define N_MIN 100
#define N_MAX 100000000 //1e9

// #define ENABLE_BOOST
// #define ENABLE_THRUST
#define ENABLE_VIENNACL
//

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef ENABLE_BOOST
// boost
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>
#endif

#ifdef ENABLE_THRUST
// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#endif

#ifdef ENABLE_VIENNACL
// ViennaCL
#define VIENNACL_WITH_CUDA
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#endif

//
//----------- Helper functions
//
template <typename T>
void printContainer(T container, const int size)
{
    std::cout << container[0];
    for (int i = 1; i < size; ++i)
        std::cout << " | " << container[i];
    std::cout << std::endl;
}

template <typename T>
void printContainer(T container, const int size, const int only)
{
    std::cout << container[0];
    for (int i = 1; i < only; ++i)
        std::cout << " | " << container[i];
    std::cout << " | ...";
    for (int i = size - only; i < size; ++i)
        std::cout << " | " << container[i];
    std::cout << std::endl;
}

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

bool check(const double *test, const double *ref, const size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        if (test[i] != ref[i])
            return false;
    }
    return true;
}

double diff_norm(const double *test, const double *ref, const size_t N)
{
    double norm = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        norm += test[i] != ref[i];
    }
    return sqrt(norm);
}
//
//----------- functions for this program
//
#ifdef ENABLE_BOOST
int boost_benchmark()
{
    namespace compute = boost::compute;
    // get default device and setup context
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    // generate random data on the host
    std::vector<float> host_vector(10000);
    std::generate(host_vector.begin(), host_vector.end(), rand);

    // create a vector on the device
    compute::vector<float> device_vector(host_vector.size(), context);

    // transfer data from the host to the device
    compute::copy(
        host_vector.begin(), host_vector.end(), device_vector.begin(), queue);

    // calculate the square-root of each element in-place
    compute::transform(
        device_vector.begin(),
        device_vector.end(),
        device_vector.begin(),
        compute::sqrt<float>(),
        queue);

    // copy values back to the host
    compute::copy(
        device_vector.begin(), device_vector.end(), host_vector.begin(), queue);

    std::cout << host_vector[0] << ", " << host_vector[1] << ", ..." << std::endl;

    return 0;
}
#endif

#ifdef ENABLE_THRUST
int thrust_benchmark(void) {

  // generate 32M random numbers on the host
  thrust::host_vector<int> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<int> d_vec= h_vec;
  // sort data on the device
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  std::cout << h_vec[0] << ", " << h_vec[1] << ", " << h_vec[2] << std::endl;

  return 0;
}
#endif

#ifdef ENABLE_VIENNACL
int viennacl_benchmark(size_t N, double x_init, double y_init, std::vector<double>& results)
{
    Timer timer;
    timer.reset();
    viennacl::vector<double> x = viennacl::scalar_vector<double>(N, x_init);
    viennacl::vector<double> y = viennacl::scalar_vector<double>(N, y_init);
    results[0] = timer.get();

    timer.reset();
    // viennacl::scalar<double> dot = viennacl::linalg::inner_prod(x+y, x-y);
    double dot = viennacl::linalg::inner_prod(x+y, x-y);
    results[1] = timer.get();

    double true_dot = (x_init + y_init) * (x_init - y_init) * N;

#ifdef COUT
    std::cout << "(x+y, x-y) = " << dot << " ?= " << true_dot << std::endl;
    std::cout << "Computation took " << results[1] << "s" << std::endl;
#endif
    timer.reset();
    results[3] = dot;
    results[2] = timer.get();

    return 0;
}
#endif

int main(int argc, char const *argv[])
{
    size_t N = 100;
    double x_init = 1., y_init = 2.;
    

#ifdef ENABLE_BOOST
    std::vector<double> results_boost(3, 0.0);
    boost_benchmark(N, x_init, y_init, results_boost);
#endif

#ifdef ENABLE_THRUST
    std::vector<double> results_thrust(3, 0.0);
    thrust_benchmark(N, x_init, y_init, results_thrust);
#endif

#ifdef ENABLE_VIENNACL
    std::vector<double> results_viennacl(4, 0.0);
    viennacl_benchmark(N, x_init, y_init, results_viennacl);
#endif

    std::cout << "Data: https://gtx1080.360252.org/2020/" << EX << "/" << CSV_NAME;

    return EXIT_SUCCESS;
}
