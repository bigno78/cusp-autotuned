#include <cusp/multiply.h>
#include <cusp/array1d.h>
#include <cusp/print.h>
#include <cusp/coo_matrix.h>
#include <cusp/io/matrix_market.h>

#include <cusp/ktt/ktt.h>

#include <cmath>
#include <type_traits>
#include <sstream>
#include <iomanip>
#include <fstream>

#include <chrono>       // chrono
#include <ostream>      // ostream

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

template<typename T, typename Mem>
auto sum(const cusp::array1d<T, Mem>& array)
{
    T total = 0;
    for (const auto& val : array)
        total += val;
    return total;
}

template<typename Matrix, typename Array>
auto run_multiply(Matrix& A, Array& x, Array& y)
{
    auto clear = [](auto& array)
    {
        for (int i = 0; i < array.size(); ++i)
            array[i] = 0;
    };

    clear(y);

    auto& tuner = cusp::ktt::get_tuner();

    auto kernel_ctx = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);

    // auto conf = tuner.CreateConfiguration(kernel_ctx.kernel_id,
    //                 { { std::string("BLOCK_SIZE"), uint64_t(1) }, });
    auto res = cusp::ktt::multiply(A, x, y);

    return res;
}

template<typename Matrix>
void load(const std::string& path, Matrix& out)
{
    cusp::io::read_matrix_market_file(out, path);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " matrix_path\n";
        return 1;
    }

    auto file = std::string(argv[1]);

    // auto A = cusp::coo_matrix<int, float, cusp::device_memory>(100, 100, 10);
    // auto A = example_mat();

    auto A = cusp::coo_matrix<int, float, cusp::device_memory>();

    cusp::io::read_matrix_market_file(A, file);

    cusp::array1d<float, cusp::device_memory> x(A.num_cols, 1);
    cusp::array1d<float, cusp::device_memory> y(A.num_rows);

    cusp::multiply(A, x, y);
    std::cout << "Reference sum: " << sum(y) << "\n";

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Reference time: " << diff << " ms\n";

    cusp::ktt::enable();

    const int COUNT = 5;
    for (int i = 0; i < COUNT; ++i)
    {
        auto res = run_multiply(A, x, y);
        std::cout << "Configuration: "
                << res.GetConfiguration().GetString() << "\n";
        // cusp::print(y);
        // print_array(y) << "\n";
        std::cout << sum(y) << "\n";
    }

    return 0;
}
