#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "timer.hpp"
// thrust
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// DEFINES
#define EX "ex8"
#define CSV_NAME "ph_data_thrust_cuda.csv"

#define COUT
#define NUM_TEST 5
#define N_MIN 10
#define N_MAX 10000000 //1e8
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
//----------- functions for this program
//

double benchmark(size_t N, double x_init, double y_init, std::vector<double>& results)
{

    Timer timer;
    timer.reset();
    thrust::host_vector<double> x(N, x_init);
    thrust::host_vector<double> y(N, y_init);

    thrust::device_vector<double> X = x;
    thrust::device_vector<double> Y = y;
    thrust::device_vector<double> TMP(N);
    thrust::device_vector<double> TMP2(N);
    results[0] = timer.get();

    double dot;

    std::vector<double> tmp(NUM_TEST, 0.0);
    for (int iter = 0; iter < NUM_TEST; iter++) {
        timer.reset(); 
        thrust::transform(X.begin(), X.end(), 
            Y.begin(), TMP.begin(), thrust::plus<double>{});
        // I tried to reuse the vector X for the result of the last transform,
        // but it did not work properly. I assume, that the reason is that these
        // are asynchronous calls that can happen in parallel, 
        // so it might happen that parts of X
        // are overwritten before the first is finished.
        // That seems weird...
        thrust::transform(X.begin(), X.end(), 
            Y.begin(), TMP2.begin(), thrust::minus<double>{});

        dot = thrust::inner_product(TMP.begin(), TMP.end(), 
                    TMP2.begin(), 0.0);
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

    return dot;
}

int main(int argc, char const *argv[])
{
    double x_init = 1., y_init = 2.;
    std::vector<double> results(4, 0.0);

    std::ofstream csv;
    std::string sep = ";";
    std::string header = "N;vec_init_time;dot_time;memcpy_time;dot_result";
    auto to_csv = [&csv, &sep] (double x) { csv << sep << x;};

    csv.open(CSV_NAME, std::fstream::out | std::fstream::trunc);
    csv << header << std::endl;
    for (size_t N = N_MIN; N < 1+N_MAX; N*=10){
#ifdef COUT
        std::cout << "N: " <<  N << std::endl;
#endif
        benchmark(N, x_init, y_init, results);
        csv << N;
        std::for_each(results.begin(), results.end(), to_csv);
        csv << std::endl;
    }

    std::cout << "Data: https://gtx1080.360252.org/2020/" << EX << "/" << CSV_NAME;

    return EXIT_SUCCESS;
}
