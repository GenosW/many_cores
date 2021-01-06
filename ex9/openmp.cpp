#include <omp.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "timer.hpp"

// DEFINES
#define EX "ex9"
#define CSV_NAME "ph_data_omp.csv"
#define COUT
#define NUM_TEST 5

#define X_VAL 1
#define Y_VAL 2

using size_t = std::size_t;

template <template <typename, typename> class Container,
          typename ValueType,
          typename Allocator = std::allocator<ValueType>>
double median(Container<ValueType, Allocator> data)
{
    size_t size = data.size();
    if (size == 0)
        return 0.;
    std::sort(data.begin(), data.end());
    size_t mid = size / 2;

    return size % 2 == 0 ? (data[mid] + data[mid - 1]) / 2 : data[mid];
};

void init_csv(std::ofstream &csv){
    std::string header = "N;k;runtime";
    csv.open(CSV_NAME, std::fstream::out | std::fstream::trunc);
    csv << header << std::endl;
}

int main()
{
    Timer timer;
    std::vector<size_t> Ks{1, 2, 3, 4, 5, 6, 7}; // pythonic way to do it... but this "program" is really "scripty" anyway...
    std::vector<size_t> Ns(Ks.size());
    size_t N_max = std::pow(10, Ks.back());

    std::vector<double> runtimes(Ks.size(), 0);

    // Init vectors once --> okay here and saves a bit of (unnecessary) time
    std::vector<double> x(N_max, X_VAL);
    std::vector<double> y(N_max, Y_VAL);
    // since map doesn't really like std::vector, we have to get the
    // underlying data pointers. Although, there seems to be a more C++-style
    // way via iterators (did not investigate that any further).
    double *xp = x.data();
    double *yp = y.data();

    size_t cnt = 0;
    for (auto &k : Ks)
    {
        size_t N = std::pow(10, k); // I know it's slow, but eh...
        Ns[cnt] = N;
        std::cout << "k: " << k << " --> N = " << N << std::endl;

        double result = 0;
        double expected = (double)N * double(X_VAL + Y_VAL) * double(X_VAL - Y_VAL);
        std::vector<double> tmp(NUM_TEST, 0.0);
        for (int iter = 0; iter < NUM_TEST; iter++)
        {
            result = 0;
            timer.reset();
    //          >distribute work on "GPU"< >as parallel for<
    #pragma omp target teams distribute parallel for \
        map(to: xp [0:N], yp [0:N]) \
        map(tofrom: result) \
        reduction(+: result)
            for (size_t i = 0; i < N; ++i)
            {
                result += (xp[i] + yp[i]) * (xp[i] - yp[i]);
            }
            tmp[iter] = timer.get();
        }
        double runtime = median(tmp);
        runtimes[cnt] = runtime;

        std::string check = (result == expected) ? "Y" : "N";
        std::cout << "Reduction result: " << result << std::endl;
        std::cout << "Expected  result: " << expected << " [" << check << "]" << std::endl;
        std::cout << "Runtime: " << runtime << std::endl;
        cnt++;
    }

    std::cout << "----------- SUMMARY -----------" << std::endl;

    std::cout << "Runtimes: " << std::endl;
    for (auto &rt: runtimes)
        std::cout << rt << std::endl;

    std::ofstream csv;
    init_csv(csv);
    std::string sep = ";";
    for (size_t idx = 0; idx < Ks.size(); ++idx)
    {
        csv << Ns[idx] << sep
            << Ks[idx] << sep
            << runtimes[idx] << std::endl;
    }
    csv.close();

    std::cout << "Data: https://gtx1080.360252.org/2020/" << EX << "/" << CSV_NAME;

    return EXIT_SUCCESS;
}
