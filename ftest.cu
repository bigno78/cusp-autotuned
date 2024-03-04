// native cusp
#include <cusp/multiply.h>
#include <cusp/array1d.h>
#include <cusp/print.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>

// ktt cusp
#include <cusp/ktt/ktt.h>

#include <chrono>       // chrono
#include <type_traits>  // is_same
#include <ostream>      // ostream
#include <string>       // stoi

using MeasureUnit = std::chrono::microseconds;

template<typename Unit>
constexpr ktt::TimeUnit chrono_to_ktt_unit()
{
    if constexpr (std::is_same_v<Unit, std::chrono::milliseconds>)
        return ktt::TimeUnit::Milliseconds;

    if constexpr (std::is_same_v<Unit, std::chrono::microseconds>)
        return ktt::TimeUnit::Microseconds;

    if constexpr (std::is_same_v<Unit, std::chrono::nanoseconds>)
        return ktt::TimeUnit::Nanoseconds;

    assert(false);
    // TODO: error if wrong
}

template<typename Unit, typename T>
std::string show_diff(T diff)
{
    std::stringstream o;
    auto count = std::chrono::duration_cast<Unit>(diff).count();

    o << count;

    if      constexpr (std::is_same_v<Unit, std::chrono::milliseconds>) o << "ms";
    else if constexpr (std::is_same_v<Unit, std::chrono::microseconds>) o << "us";
    else if constexpr (std::is_same_v<Unit, std::chrono::nanoseconds>)  o << "ns";
    else
        return "invalid unit type";

    return o.str();
}

template<typename Func>
auto measure_time(Func func)
{
    auto start = std::chrono::steady_clock::now();

    auto res = func();

    auto end = std::chrono::steady_clock::now();

    std::cout << "Time: "
              << show_diff<MeasureUnit>(end - start)
              << "\n";

    return res;
}

template<typename Array>
std::ostream& print_array(const Array& array, std::ostream& o = std::cout)
{
    o << "< ";
    const char* del = "";
    for (const auto& val : array)
    {
        o << del << val;
        del = ", ";
    }
    return o << " >";
}

// template<typename T, typename Mem>
// auto sum(const cusp::array1d<T, Mem>& array)
// {
//     T total = 0;
//     for (const auto& val : array)
//         total += val;
//     return total;
// }

template<typename T, typename Mem>
auto sparse_sum(const cusp::array1d<T, Mem>& array)
{
    T total = 0;
    for (std::size_t i = 0; i < array.size(); i = i *  2 + 1)
        total += array[i];
    return total;
}
#include <thrust/fill.h>
template<typename Matrix, typename Array>
auto run_multiply(Matrix& A, Array& x, Array& y)
{
    auto clear = [](auto& array)
    {
        for (int i = 0; i < array.size(); ++i)
            array[i] = 0;
    };

    // clear(y);
    thrust::fill(y.begin(), y.end(), 1);

    auto& tuner = cusp::ktt::get_tuner();
    tuner.SetTimeUnit(chrono_to_ktt_unit<MeasureUnit>());

    auto kernel_ctx = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);

    // auto conf = tuner.CreateConfiguration(kernel_ctx.kernel_id,
    //                 { { std::string("BLOCK_SIZE"), uint64_t(1) }, });

    auto res = measure_time([&](){ return cusp::ktt::multiply(A, x, y); });

    return res;
}

template<typename Matrix>
void load(const std::string& path, Matrix& out)
{
    cusp::io::read_matrix_market_file(out, path);
}


template<template<class T1, class T2, class T3> typename MatrixFmt>
void run(const MatrixFmt<int, float, cusp::device_memory>& A,
         const int COUNT)
{
    cusp::array1d<float, cusp::device_memory> x(A.num_cols, 1);
    cusp::array1d<float, cusp::device_memory> y(A.num_rows);
    cusp::array1d<float, cusp::device_memory> ref_y(A.num_rows);

    cusp::ktt::disable();

    // HEAT UP
    const int HEAT_UP_COUNT = 5;
    for (int i = 0; i < HEAT_UP_COUNT; ++i)
        cusp::multiply(A, x, ref_y);

    measure_time([&]()
    {
        cusp::multiply(A, x, ref_y);
        return 0;
    });


    std::cout << "Reference sum: " << sparse_sum(ref_y) << "\n";

    // print_array(ref_y) << "\n";

    cusp::ktt::enable();

    for (int i = 0; i < COUNT; ++i)
    {
        auto res = run_multiply(A, x, y);
        // auto res = measure_time([&](){ return run_multiply(A, x, y); });
        std::cout << "Configuration: "
                  << res.GetConfiguration().GetString() << "\n";
        // cusp::print(y);
        // print_array(y) << "\n";
        std::cout << "Chk sum: " << sparse_sum(y) << "\n";
        std::cout << (y == ref_y) << std::endl;
    }
}


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " coo|csr matrix_path [count=5]\n";
        return 1;
    }

    auto fmt = std::string(argv[1]);
    auto file = std::string(argv[2]);

    int COUNT = 5;
    if (argc >= 4)
        COUNT = std::stoi(argv[3]);

    std::cout << "THRUST_VERSION=" << THRUST_VERSION << "\n";

    if (fmt == "coo")
    {
        auto A = cusp::coo_matrix<int, float, cusp::device_memory>();
        cusp::io::read_matrix_market_file(A, file);
        std::cout << "COO\n";
        run(A, COUNT);
    }
    else if (fmt == "csr")
    {
        auto A = cusp::csr_matrix<int, float, cusp::device_memory>();
        cusp::io::read_matrix_market_file(A, file);
        std::cout << "CSR\n";
        run(A, COUNT);
    }

    return 0;
}
