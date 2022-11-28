#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/gallery/poisson.h>

#include <cusp/ktt/ktt.h>
#include <cusp/ktt/matrix_generation.h>

#include <chrono>
#include <cmath>
#include <type_traits>
#include <sstream>
#include <iomanip>

namespace krn = std::chrono;


struct Timer
{
    Timer() : start_(krn::steady_clock::now()) { }

    void restart() { start_ = krn::steady_clock::now(); }

    void print_elapsed(const std::string& label)
    {
        auto duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start_).count();
        std::cout << label << ": " << duration << "\n";
    }

    void checkpoint(const std::string& label)
    {
        print_elapsed(label);
        restart();
    }

private:
    krn::steady_clock::time_point start_;

};


struct ConversionTimer
{
    cusp::dia_matrix<int, float, cusp::host_memory> m;

    ConversionTimer() {
        m = cusp::ktt::make_diagonal_symmetric_matrix(4096, 4096, 4, 512);
    }

    ConversionTimer(const ConversionTimer& o) {
        std::cout << "COPY!\n";
    }

    template<typename Matrix1, typename Matrix2>
    void operator()()
    {
        using Format1 = typename Matrix1::format;
        using Format2 = typename Matrix2::format;

        Matrix1 m1 = m;

        auto start = krn::steady_clock::now();
        Matrix2 m2 = m1;
        auto duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start).count();

        std::cout << NAMEOF_TYPE(Format1) << " to " << NAMEOF_TYPE(Format2) << ": " << duration << "\n";
    }
};


template<typename... Types>
struct type_list
{
    static constexpr bool is_empty = true;
};

template<
    typename First,
    typename... Args
>
struct type_list<First, Args ...>
{
    using head = First;
    using tail = type_list<Args ...>;

    static constexpr bool is_empty = false;
};


template<
    typename T,
    typename List,
    typename Op
>
void for_each_type(Op&& op)
{
    if constexpr (!List::is_empty)
    {
        if (!std::is_same_v<typename List::head, T>)
        {
            op.template operator()<T, typename List::head>();
        }
        for_each_type<T, typename List::tail>(std::forward<Op>(op));
    }
}

template<typename List1, typename List2, typename Op>
void for_each_pair(Op&& op)
{
    if constexpr (!List1::is_empty && !List2::is_empty)
    {
        for_each_type<typename List1::head, List2>(std::forward<Op>(op));
        for_each_pair<typename List1::tail, List2>(std::forward<Op>(op));
    }
}

template<typename List, typename Op>
void for_each_pair(Op&& op)
{
    for_each_pair<List, List>(std::forward<Op>(op));
}

template<typename IndexType, typename ValueType, typename MemorySpace>
using format_list = type_list<
    cusp::dia_matrix<IndexType, ValueType, MemorySpace>,
    cusp::csr_matrix<IndexType, ValueType, MemorySpace>,
    cusp::coo_matrix<IndexType, ValueType, MemorySpace>,
    cusp::ell_matrix<IndexType, ValueType, MemorySpace>
>;


void print_ktt_status(const ::ktt::ResultStatus& status) {
    switch (status)
    {
    case ::ktt::ResultStatus::Ok:
        std::cout << "OK\n";
        break;
    case ::ktt::ResultStatus::CompilationFailed:
        std::cout << "COMPILATION FAILED\n";
        break;
    case ::ktt::ResultStatus::ComputationFailed:
        std::cout << "COMPUTATION FAILED\n";
        break;
    case ::ktt::ResultStatus::DeviceLimitsExceeded:
        std::cout << "DEVICE LIMIT EXCEEDED\n";
        break;
    case ::ktt::ResultStatus::ValidationFailed:
        std::cout << "VALIDATION FAILED\n";
        break;
    }
}

template<typename IndexType, typename ValueType>
cusp::dia_matrix<IndexType, ValueType, cusp::host_memory>
convert(const cusp::coo_matrix<IndexType, ValueType, cusp::host_memory>& matrix) {
    using DiaMatrix = cusp::dia_matrix<IndexType, ValueType, cusp::host_memory>;

    cusp::array1d<int, cusp::host_memory> diag_map(matrix.num_rows + matrix.num_cols, 0);

    auto start = krn::steady_clock::now();
    for (int i = 0; i < matrix.num_entries; ++i) {
        int diag_idx = matrix.column_indices[i] - matrix.row_indices[i] + matrix.num_rows - 1;
        diag_map[diag_idx] = 1;
    }
    auto duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start).count();
    std::cout << "First loop:  " << duration << "\n";

    typename DiaMatrix::diagonal_offsets_array_type offsets;
    offsets.reserve(512); // arbitrary number

    start = krn::steady_clock::now();
    int diag_count = 0;
    for (int i = 0; i < diag_map.size(); ++i) {
        if (diag_map[i] != 0) {
            diag_map[i] = diag_count;
            offsets.push_back(i - matrix.num_rows + 1);
            diag_count++;
        }
    }
    duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start).count();
    std::cout << "Third loop:  " << duration << "\n";


    //int pitch = matrix.num_rows % 2 == 0 ? matrix.num_rows + 1 : matrix.num_rows;
    int pitch = matrix.num_rows + 64/sizeof(float);
    typename DiaMatrix::values_array_type values(matrix.num_rows, diag_count, ValueType(0), pitch);
    //std::cout << values.pitch << "\n";

    start = krn::steady_clock::now();
    for (int i = 0; i < matrix.num_entries; ++i) {
        int diag_idx = matrix.column_indices[i] - matrix.row_indices[i] + matrix.num_rows - 1;
        int values_array_idx = diag_map[diag_idx];
        values(matrix.row_indices[i], values_array_idx) = matrix.values[i];
    }
    duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start).count();
    std::cout << "Second loop: " << duration << "\n";

    DiaMatrix res;

    res.num_rows = matrix.num_rows;
    res.num_cols = matrix.num_cols;
    res.num_entries = matrix.num_entries;

    res.diagonal_offsets.swap(offsets);
    res.values.swap(values);

    return res;
}

template<typename IndexType, typename ValueType>
cusp::dia_matrix<IndexType, ValueType, cusp::host_memory>
convert2(const cusp::coo_matrix<IndexType, ValueType, cusp::host_memory>& matrix) {
    using DiaMatrix = cusp::dia_matrix<IndexType, ValueType, cusp::host_memory>;

    Timer timer;

    cusp::array1d<int, cusp::host_memory> diag_map(matrix.num_rows + matrix.num_cols, 0);

    timer.checkpoint("init map");

    for (int i = 0; i < matrix.num_entries; ++i) {
        int diag_idx = matrix.column_indices[i] - matrix.row_indices[i] + matrix.num_rows - 1;
        diag_map[diag_idx] = 1;
    }

    timer.checkpoint("find diags");

    typename DiaMatrix::diagonal_offsets_array_type offsets;
    offsets.reserve(512); // arbitrary number

    int diag_count = 0;
    for (int i = 0; i < diag_map.size(); ++i) {
        if (diag_map[i] != 0) {
            diag_map[i] = diag_count;
            offsets.push_back(i - matrix.num_rows + 1);
            diag_count++;
        }
    }

    timer.checkpoint("fill map");


    //int pitch = matrix.num_rows % 2 == 0 ? matrix.num_rows + 1 : matrix.num_rows;
    typename DiaMatrix::values_array_type values(diag_count, matrix.num_rows, ValueType(0));
    //std::cout << values.pitch << "\n";

    timer.checkpoint("init values");

    for (int i = 0; i < matrix.num_entries; ++i) {
        int diag_idx = matrix.column_indices[i] - matrix.row_indices[i] + matrix.num_rows - 1;
        int values_array_idx = diag_map[diag_idx];
        values(values_array_idx, matrix.row_indices[i]) = matrix.values[i];
    }

    timer.checkpoint("fill values");

    DiaMatrix res;

    res.num_rows = matrix.num_rows;
    res.num_cols = matrix.num_cols;
    res.num_entries = matrix.num_entries;

    res.diagonal_offsets.swap(offsets);
    res.values.swap(values);

    return res;
}

template<typename IndexType, typename ValueType>
bool equals(const cusp::dia_matrix<IndexType, ValueType, cusp::host_memory>& A,
            const cusp::dia_matrix<IndexType, ValueType, cusp::host_memory>& B)
{
    return A.values == B.values && A.diagonal_offsets == B.diagonal_offsets;
}

std::string size_str(uint64_t bytes)
{
    float size = bytes;

    auto get_str = [](float x, const char* units)
    {
        std::stringstream s;
        s << std::setprecision(3) << std::fixed << x << units;
        return s.str();
    };

    if (size < 1024)
    {
        return get_str(size, " B");
    }

    size /= 1024;
    if (size < 1024)
    {
        return get_str(size, " KB");
    }

    size /= 1024;
    if (size < 1024)
    {
        return get_str(size, " MB");
    }

    size /= 1024;
    return get_str(size, " GB");
}

std::string counter_str(const ::ktt::KernelProfilingCounter& counter)
{
    switch (counter.GetType()) {
        case ::ktt::ProfilingCounterType::Double:
            return std::to_string(counter.GetValueDouble());
        case ::ktt::ProfilingCounterType::Int:
            return std::to_string(counter.GetValueInt());
        case ::ktt::ProfilingCounterType::Percent:
            return std::to_string(counter.GetValueDouble()) + " %";
        case ::ktt::ProfilingCounterType::Throughput:
            return std::to_string(counter.GetValueUint()) + " bytes/sec\n";
        case ::ktt::ProfilingCounterType::UnsignedInt:
            return std::to_string(counter.GetValueUint());
        case ::ktt::ProfilingCounterType::UtilizationLevel:
            return std::to_string(counter.GetValueUint());
    }

    return "";
}

size_t dia_matrix_size(size_t num_rows, size_t num_cols, size_t num_diags)
{
    return sizeof(int)*num_diags + num_rows*num_diags*sizeof(float);
}

size_t dia_problem_size(size_t num_rows, size_t num_cols, size_t num_diags)
{
    return sizeof(int)*num_diags
        + num_rows*num_diags*sizeof(float)
        + num_rows*sizeof(float)
        + num_cols*sizeof(float);
}

template<typename IndexT,
         typename ValueT,
         typename MemorySpace>
size_t min_read_bytes(const cusp::dia_matrix<IndexT, ValueT, MemorySpace>& A)
{
    // Assume that each column has at least one diagonal passing through it.
    size_t result = A.num_cols * sizeof(ValueT);

    for (IndexT offset : A.diagonal_offsets)
    {
        IndexT start_row = offset >= 0 ? 0 : -offset;
        IndexT start_col = offset >= 0 ? offset : 0;
        IndexT end_row = std::min(A.num_cols - start_col, A.num_rows - start_row);

        // Assume that global reads are done in 32 byte transactions that must be aligned.
        IndexT elems_per_read = 32 / sizeof(ValueT);
        IndexT adjusted_start_row = start_row - (start_row % elems_per_read);
        IndexT adjusted_end_row = end_row + elems_per_read - (end_row % elems_per_read);

        result += (adjusted_end_row - adjusted_start_row)*sizeof(ValueT);
    }

    return result;
}

std::vector<::ktt::KernelProfilingCounter>
profile_multiply(const cusp::dia_matrix<int, float, cusp::device_memory>& A,
                 const cusp::array1d<float, cusp::device_memory>& x,
                 cusp::array1d<float, cusp::device_memory>& y,
                 const ::ktt::KernelConfiguration& conf,
                 const std::vector<std::string>& counters)
{
    auto& tuner = cusp::ktt::get_tuner();

    auto kernel_ctx = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);

    tuner.SetProfilingCounters(counters);
    tuner.SetProfiledDefinitions(kernel_ctx.kernel_id, kernel_ctx.definition_ids);
    tuner.SetProfiling(true);

    ::ktt::KernelResult kernel_result;
    do {
        kernel_result = cusp::ktt::multiply(A, x, y, conf, true);
    } while(kernel_result.HasRemainingProfilingRuns());

    for (const auto& computation_result : kernel_result.GetResults())
    {
        if (!computation_result.HasProfilingData())
        {
            std::cout << "No profiling data!\n";
            continue;
        }

        const auto& profiling_data = computation_result.GetProfilingData();
        if (!profiling_data.IsValid())
        {
            std::cout << "profiling data invalid\n";
        }
        else
        {
            return profiling_data.GetCounters();
        }
    }

    return {};
}

size_t get_actual_read_bytes(::ktt::Tuner& tuner,
                             const cusp::dia_matrix<int, float, cusp::device_memory>& A,
                             const cusp::array1d<float, cusp::device_memory>& x,
                             cusp::array1d<float, cusp::device_memory>& y,
                             const ::ktt::KernelConfiguration& conf)
{
    const auto& counters = profile_multiply(A, x, y, conf, {"dram_read_bytes"});

    if (counters.empty()) {
        throw std::runtime_error("Profiling failed.");
    }

    return counters[0].GetValueUint();
}

size_t get_actual_read_bytes(::ktt::Tuner& tuner,
                             const cusp::dia_matrix<int, float, cusp::host_memory>& host_matrix)
{
    cusp::dia_matrix<int, float, cusp::device_memory> A = host_matrix;
    cusp::array1d<float, cusp::device_memory> x(A.num_cols, 1);
    cusp::array1d<float, cusp::device_memory> y(A.num_rows);

    auto kernel_ctx = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);
    auto conf = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(1) },
                                                                  { std::string("PREFETCH_FACTOR"), uint64_t(0) } });

    return get_actual_read_bytes(tuner, A, x, y, conf);
}

int get_max_size(int num_diags, size_t global_size)
{
    // Assume we have a square matrix of size n.
    //
    // On the gpu we need:
    //   - offsets   -> num_diags * sizeof(int) bytes
    //   - diagonals -> num_diags * n * sizeof(float) bytes
    //   - x         ->             n * sizeof(float) bytes
    //   - y         ->             n * sizeof(float) bytes
    //
    // It must hold:
    //
    //   global_size >= num_diags*sizeof(int) + (num_diags + 2)*n*sizeof(float)
    //
    // So:
    //
    //   n <= (global_size - num_diags*sizeof(int)) / (num_diags + 2)*sizeof(float)
    //
    return (global_size - num_diags*sizeof(int)) / ((num_diags + 2)*sizeof(float));
}

void test_poisson(::ktt::Tuner& tuner,
                  const cusp::dia_matrix<int, float, cusp::host_memory>& host_matrix,
                  const std::string& name)
{
    auto expected_bytes = min_read_bytes(host_matrix);
    auto actual_bytes = get_actual_read_bytes(tuner, host_matrix);
    std::cout << "  " << name << ": " << size_str(actual_bytes) << " vs " << size_str(expected_bytes) << "\n";
}

void test_poisson_sizes(::ktt::Tuner& tuner, size_t global_size)
{
    std::cout << "Testing poisson\n";

    int n = get_max_size(5, global_size);
    int grid_size = std::sqrt(n);
    cusp::dia_matrix<int, float, cusp::host_memory> dia_host_matrix;
    cusp::gallery::poisson5pt(dia_host_matrix, grid_size, grid_size);
    test_poisson(tuner, dia_host_matrix, "5pt");

    n = get_max_size(9, global_size);
    grid_size = std::sqrt(n);
    cusp::gallery::poisson9pt(dia_host_matrix, grid_size, grid_size);
    test_poisson(tuner, dia_host_matrix, "9pt");

    n = get_max_size(27, global_size);
    grid_size = std::cbrt(n);
    cusp::gallery::poisson27pt(dia_host_matrix, grid_size, grid_size, grid_size);
    test_poisson(tuner, dia_host_matrix, "27pt");

    n = get_max_size(7, global_size);
    grid_size = std::cbrt(n);
    cusp::gallery::poisson7pt(dia_host_matrix, grid_size, grid_size, grid_size);
    test_poisson(tuner, dia_host_matrix, "7pt");


    // std::cout << "  Memory requirements: "
    //           << size_str(dia_problem_size(dia_host_matrix.num_rows,
    //                                        dia_host_matrix.num_cols,
    //                                        dia_host_matrix.diagonal_offsets.size()))
    //           << "\n";
}

void test_poisson_7pt(::ktt::Tuner& tuner, size_t global_limit)
{
    std::cout << "Testing different sizes of poisson 7pt problem (actual vs min)\n";

    size_t mb = (size_t(1) << 20);
    cusp::dia_matrix<int, float, cusp::host_memory> host_matrix;

    for (size_t mem = 512*mb; mem <= global_limit; mem += 512*mb)
    {
        int n = get_max_size(7, mem);
        int grid_size = std::cbrt(n);

        cusp::gallery::poisson7pt(host_matrix, grid_size, grid_size, grid_size);

        auto expected_bytes = min_read_bytes(host_matrix);
        auto actual_bytes = get_actual_read_bytes(tuner, host_matrix);
        float ratio = float(actual_bytes)/expected_bytes;

        std::cout << "  "
                  << size_str(mem) << " (" << grid_size << ") -> "
                  << size_str(actual_bytes) << " vs "
                  << size_str(expected_bytes) << " ("
                  << ratio << ")\n";
    }
}

void test_l2()
{
    auto& tuner = cusp::ktt::get_tuner();

    tuner.SetLoggingLevel(::ktt::LoggingLevel::Off);

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess)
    {
        std::cout << "Failed to read device properties.\n";
        return;
    }

    size_t max_bytes = std::min(5*(size_t(1) << 30), (size_t)prop.totalGlobalMem - 1024*(size_t(1) << 20));
    size_t max_floats = max_bytes/sizeof(float);

    std::cout << "L2 size: " << size_str(prop.l2CacheSize) << "\n";
    std::cout << "Global memory available: " << size_str(prop.totalGlobalMem) << "\n";
    std::cout << "Global limit: " << size_str(max_bytes) << "\n";

    test_poisson_sizes(tuner, max_bytes);
    test_poisson_7pt(tuner, max_bytes);

    // std::cout << "Using size: " << size_str(max_bytes) << "\n";

    // // const size_t n = (1u << 20)*4;

    // for (int dist = 1; dist <= 1024; dist *= 2)
    // {
    //     for (int diags = 4; diags <= 1024; diags *= 2)
    //     {
    //         int actual_diags = diags + 1; // to make it nicely symmetric
    //         int n = max_floats/(actual_diags + 1);

    //         if (dist*diags > n)
    //         {
    //             std::cout << "skipping\n";
    //             continue;
    //         }

    //         auto dia_host_matrix = cusp::ktt::make_diagonal_symmetric_matrix(n, n, dist, actual_diags);

    //         cusp::dia_matrix<int, float, cusp::device_memory> A = dia_host_matrix;
    //         cusp::array1d<float, cusp::device_memory> x(A.num_cols, 1);
    //         cusp::array1d<float, cusp::device_memory> y(A.num_rows);

    //         auto kernel_ctx = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);
    //         auto conf = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(1) },
    //                                                                       { std::string("PREFETCH_FACTOR"), uint64_t(0) } });

    //         auto expected_bytes = min_read_bytes(A, x);
    //         auto actual_bytes = get_actual_read_bytes(A, x, y, conf);

    //         float ration = float(actual_bytes)/expected_bytes;

    //         std::cout << "(" << size_str(n) << ", " << dist << ", " << actual_diags << ") -> " << ration << "\n";

    //         // std::cout << "Matrix size: " << size_str(dia_matrix_size(n, n, actual_diags)) << "\n";
    //         // std::cout << "Expected: " << size_str(expected_bytes) << "\n";
    //         // std::cout << "Actual:   " << size_str(actual_bytes) << "\n";
    //         // std::cout << "------------------------------------------------\n";
    //     }
    // }

    // auto dia_host_matrix = cusp::ktt::make_diagonal_symmetric_matrix(n, n, 1, 11);

    // cusp::dia_matrix<int, float, cusp::device_memory> A = dia_host_matrix;
    // cusp::array1d<float, cusp::device_memory> x(A.num_cols, 1);
    // cusp::array1d<float, cusp::device_memory> y(A.num_rows);

    // auto kernel_ctx = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);
    // auto conf = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(1) } });

    // auto expected_bytes = min_read_bytes(A, x);
    // auto actual_bytes = get_actual_read_bytes(A, x, y, conf);

    // std::cout << "Expected: " << size_str(expected_bytes) << "\n";
    // std::cout << "Actual:   " << size_str(actual_bytes) << "\n";
}

void test_matrix()
{
    auto& tuner = cusp::ktt::get_tuner();

    const int n = 8*(1u << 20);
    //const int n = (1u << 20)/2;

    cusp::dia_matrix<int, float, cusp::host_memory> dia_host_matrix;
    cusp::gallery::poisson7pt(dia_host_matrix, 390, 390, 390);
    // auto dia_host_matrix = cusp::ktt::make_diagonal_symmetric_matrix(n, n, 1, 32);

    cusp::dia_matrix<int, float, cusp::device_memory> A = dia_host_matrix;
    cusp::array1d<float, cusp::device_memory> x(A.num_cols, 1);
    cusp::array1d<float, cusp::device_memory> y(A.num_rows);

    // cusp::io::write_matrix_market_file(dia_host_matrix, "matrix.mtx");

    //auto dia_host_matrix2 = cusp::ktt::make_diagonal_symmetric_matrix(n, n, 1024, 11);
    //cusp::dia_matrix<int, float, cusp::device_memory> A2 = dia_host_matrix;

    auto kernel_ctx = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);

    // tuner.SetProfiling(true);
    // tuner.SetProfiledDefinitions(kernel_ctx.kernel_id, kernel_ctx.definition_ids);
    // tuner.SetProfilingCounters({
    //     "dram_read_bytes"
    //     //"dram_write_bytes",
    //     //"l2_global_load_bytes"
    // });

    //cusp::ktt::tune(A2, x, y);
    //cusp::ktt::tune(A, x, y);

    // auto conf1 = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(0) } });
    // cusp::ktt::multiply(A, x, y, conf1);

    auto conf2 = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(1) },
                                                                   { std::string("PREFETCH_FACTOR"), uint64_t(0) } });
    size_t non_cached = get_actual_read_bytes(tuner, A, x, y, conf2);

    conf2 = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(2) },
                                                              { std::string("PREFETCH_FACTOR"), uint64_t(0) } });
    size_t cached = get_actual_read_bytes(tuner, A, x, y, conf2);

    std::cout << "non_cached: " << size_str(non_cached) << "\n";
    std::cout << "cached:     " << size_str(cached) << "\n";

    //std::cout << tuner.GetPtxSource(kernel_ctx.kernel_id, kernel_ctx.definition_ids[0], conf2) << "\n";

    // cusp::ktt::tune(A, x, y);

    // cusp::ktt::multiply(A, x, y, conf2);
    // std::cout << size_str(get_actual_read_bytes(A, x, y, conf2)) << "\n";

    //auto expected_bytes = min_read_bytes(A, x);
    //std::cout << "Expected: " << size_str(expected_bytes) << "\n";

    //for (int i = 0; i < 10; ++i) {
    //    std::cout << y[i] << "\n";
    //}

    //std::cout << size_str(get_actual_read_bytes(A, x, y, conf2)) << "\n";
    //std::cout << size_str(get_actual_read_bytes(A, x, y, conf2)) << "\n";

    // auto conf3 = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(2) } });
    // cusp::ktt::multiply(A, x, y, conf3);
}

int main(void)
{
    auto& tuner = cusp::ktt::get_tuner();
    tuner.SetTimeUnit(::ktt::TimeUnit::Microseconds);

    test_l2();
    // test_matrix();

    // auto orig = cusp::ktt::make_diagonal_symmetric_matrix(4096, 4096, 4, 256);
    // cusp::coo_matrix<int, float, cusp::device_memory> A = orig;
    // cusp::coo_matrix<int, float, cusp::host_memory> B = orig;

    // auto start = krn::steady_clock::now();
    // cusp::dia_matrix<int, float, cusp::host_memory> B1 = convert(B);
    // auto duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start).count();
    // std::cout << "ME host:     " << duration << "\n\n";

    // start = krn::steady_clock::now();
    // cusp::dia_matrix<int, float, cusp::device_memory> B2 = convert(A);
    // duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start).count();
    // std::cout << "ME device:   " << duration << "\n\n";

    // start = krn::steady_clock::now();
    // cusp::dia_matrix<int, float, cusp::device_memory> B3 = A;
    // duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start).count();
    // std::cout << "cusp device: " << duration << "\n\n";

    // cusp::dia_matrix<int, float, cusp::host_memory> D = convert(A);
    // if (!equals(D, orig)) {
    //     std::cout << "RIIIIIIIP!!!!\n";
    // }

    // auto start = krn::steady_clock::now();
    // cusp::dia_matrix<int, float, cusp::host_memory> B1 = convert(A);
    // auto duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start).count();
    // std::cout << "ME:         " << duration << "\n\n";

    // start = krn::steady_clock::now();
    // cusp::dia_matrix<int, float, cusp::host_memory> B2 = convert2(A);
    // duration = krn::duration_cast<krn::microseconds>(krn::steady_clock::now() - start).count();
    // std::cout << "Transposed: " << duration << "\n\n";

    // if (!equals(B1, B2)) {
    //     std::cout << "RIIIIIIP!!!\n";
    // }

    // using list = format_list<int, float, cusp::host_memory>;
    // ConversionTimer timer;
    // for_each_pair<list>(timer);

    // cudaDeviceProp prop;
    // cudaError_t err = cudaGetDeviceProperties(&prop, 0);

    // if (err == cudaSuccess) {
    //     std::cout << "Max warps per sm: " << prop.maxThreadsPerMultiProcessor << "\n";
    // } else {
    //     std::cout << "Failed to retrieve device properties\n";
    // }

    // {
    //     const int n = (1u << 20);
    //     //const int n = (1u << 20)/2;

    //     //auto dia_host_matrix = cusp::ktt::make_diagonal_symmetric_matrix(n, n, 256, 1024);
    //     auto dia_host_matrix = cusp::ktt::make_diagonal_symmetric_matrix(n, n, 1, 11);
    //     cusp::dia_matrix<int, float, cusp::device_memory> A = dia_host_matrix;

    //     // cusp::io::write_matrix_market_file(dia_host_matrix, "matrix.mtx");

    //     cusp::array1d<float, cusp::device_memory> x(A.num_cols, 1);
    //     cusp::array1d<float, cusp::device_memory> y(A.num_rows);

    //     //auto dia_host_matrix2 = cusp::ktt::make_diagonal_symmetric_matrix(n, n, 1024, 11);
    //     //cusp::dia_matrix<int, float, cusp::device_memory> A2 = dia_host_matrix;

    //     auto kernel_ctx = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);

    //     //cusp::ktt::tune(A2, x, y);
    //     //cusp::ktt::tune(A, x, y);

    //     // auto conf1 = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(0) } });
    //     // cusp::ktt::multiply(A, x, y, conf1);

    //     auto conf2 = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(1) } });
    //     //std::cout << tuner.GetPtxSource(kernel_ctx.kernel_id, kernel_ctx.definition_ids[0], conf2) << "\n";
    //     cusp::ktt::multiply(A, x, y, conf2);

    //     // auto conf3 = tuner.CreateConfiguration(kernel_ctx.kernel_id, { { std::string("KERNEL_TYPE"), uint64_t(2) } });
    //     // cusp::ktt::multiply(A, x, y, conf3);
    // }

    // {
    //     const int n = 1u << 15;
    //     auto coo_host_matrix = make_diagonal_symmetric_matrix(n, n, 512, 16);
    //     cusp::coo_matrix<int, float, cusp::device_memory> coo_device_matrix = coo_host_matrix;
    //     cusp::dia_matrix<int, float, cusp::device_memory> A = coo_host_matrix;

    //     cusp::array1d<float, cusp::device_memory> x(coo_host_matrix.num_cols, 1);
    //     cusp::array1d<float, cusp::device_memory> y_ref(coo_host_matrix.num_rows);
    //     cusp::array1d<float, cusp::device_memory> y(coo_host_matrix.num_rows);

    //     cusp::multiply(coo_device_matrix, x, y_ref);

    //     auto& tuner = cusp::ktt::get_tuner();
    //     auto kernel_ctx = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);

    //     // std::cout << tuner.GetPtxSource(kernel_ctx.kernel_id, kernel_ctx.definition_ids[0]) << "\n";

    //     tuner.SetProfiling(true);
    //     tuner.SetProfiledDefinitions(kernel_ctx.kernel_id, kernel_ctx.definition_ids);

    //     tuner.SetProfilingCounters({
    //         "tex_cache_hit_rate",
    //         "tex_cache_transactions",
    //         "l1_cache_global_hit_rate",
    //         "l2_l1_read_hit_rate",
    //         "l2_l1_read_transactions",
    //         "l2_read_transactions",
    //         "l2_write_transactions",
    //         "dram_read_transactions",
    //         "dram_write_transactions",
    //         "dram_read_bytes",
    //         "dram_write_bytes"
    //     });

    //     auto result = cusp::ktt::multiply(A, x, y, {}, true);

    //     for (const auto& res : result.GetResults()) {
    //         if (!res.HasProfilingData()) {
    //             std::cout << "No profiling data!\n";
    //             continue;
    //         }
    //         const auto& prof = res.GetProfilingData();
    //         if (!prof.IsValid()) {
    //             std::cout << "profiling data invalid\n";
    //         } else {
    //             for (const auto& counter : prof.GetCounters()) {
    //                 std::cout << counter.GetName() << " = ";
    //                 switch (counter.GetType()) {
    //                     case ::ktt::ProfilingCounterType::Double:
    //                         std::cout << counter.GetValueDouble() << "\n";
    //                         break;
    //                     case ::ktt::ProfilingCounterType::Int:
    //                         std::cout << counter.GetValueInt() << "\n";
    //                         break;
    //                     case ::ktt::ProfilingCounterType::Percent:
    //                         std::cout << counter.GetValueDouble() << " %\n";
    //                         break;
    //                     case ::ktt::ProfilingCounterType::Throughput:
    //                         std::cout << counter.GetValueUint() << " bytes/sec\n";
    //                         break;
    //                     case ::ktt::ProfilingCounterType::UnsignedInt:
    //                         std::cout << counter.GetValueUint() << "\n";
    //                         break;
    //                     case ::ktt::ProfilingCounterType::UtilizationLevel:
    //                         std::cout << counter.GetValueUint() << "\n";
    //                         break;
    //                 }
    //             }
    //         }
    //     }

    //     if (y_ref != y) {
    //         std::cout << "SECOND DIFF!\n";
    //     }
    // }

    //cusp::ktt::multiply(A, x, y);
    //std::cout << y[0] << "\n";

    //std::cout << y_ref_host.size() << "\n";
    //std::cout << y.size() << "\n";

    return 0;
}
