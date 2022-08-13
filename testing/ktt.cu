#include <unittest/unittest.h>

#include <cusp/linear_operator.h>
#include <cusp/gallery/poisson.h>
#include <cusp/gallery/random.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/permutation_matrix.h>

#include <cusp/multiply.h>

#include <cusp/ktt/ktt.h>
#include <cusp/ktt/matrix_generation.h>

#include <iostream>
#include <sstream>
#include <memory>


#define DECLARE_KTT_SPARSE_FORMAT_UNITTEST(VTEST, Fmt, fmt)              \
    void VTEST##Ktt##Fmt##Matrix(void)                                   \
    {                                                                    \
        VTEST< cusp::fmt##_matrix<int, float, cusp::device_memory> >();  \
    }                                                                    \
    DECLARE_UNITTEST(VTEST##Ktt##Fmt##Matrix);

#define DECLARE_KTT_UNITTEST(VTEST) \
    DECLARE_KTT_SPARSE_FORMAT_UNITTEST(VTEST,Dia,dia)


struct UnitTestStopCondition : ::ktt::StopCondition
{

    bool IsFulfilled() const override
    {
        return failed_ || explored_configurations_ == total_configurations_;
    }

    void Initialize(const uint64_t configurationsCount) override
    {
        total_configurations_ = configurationsCount;
        explored_configurations_ = 0;
        failed_ = false;
    }

    void Update(const ::ktt::KernelResult& result) override
    {
        failed_ = failed_ || !result.IsValid();
        explored_configurations_++;
    }

    std::string GetStatusString() const override
    {
        if (failed_) {
            return "Encountered failing configuration";
        }
        return "No failing configuration encountered. Explored configurations: "
               + std::to_string(explored_configurations_) + " / " + std::to_string(total_configurations_);
    }

private:
    bool failed_ = false;
    uint64_t total_configurations_ = 0;
    uint64_t explored_configurations_ = 0;

};


void assert_tunning_results_valid(const std::vector<::ktt::KernelResult>& results,
                         const std::string& ktt_logs,
                         const std::string& arg_name,
                         const std::string& filename = "unknown",
                         int lineno = -1)
{
    bool failed = false;
    unittest::UnitTestFailure f;

    for (const auto& result : results) {
        if (result.IsValid()) {
            continue;
        }

        std::string reason = "";

        switch (result.GetStatus()) {
            case ::ktt::ResultStatus::Ok:
                continue;
            case ::ktt::ResultStatus::CompilationFailed:
                reason = "CompilationFailed";
                break;
            case ::ktt::ResultStatus::ComputationFailed:
                reason = "ComputationFailed";
                break;
            case ::ktt::ResultStatus::DeviceLimitsExceeded:
                reason = "DeviceLimitsExceeded";
                break;
            case ::ktt::ResultStatus::ValidationFailed:
                reason = "ValidationFailed";
                break;
        }

        failed = true;

        f << "[" << filename << ":" << lineno << "] " << result.GetKernelName() << ": ";
        f << "Encountered an error: " << reason << "\n\n";
        f << "On matrix: " << arg_name << "\n\n";

        f << "In configuration:\n";

        for (auto parameter : result.GetConfiguration().GetPairs()) {
            f << "  " << parameter.GetString() << "\n";
        }

        f << "\n";
    }

    if (failed) {
        f << "Logs:\n";
        f << ktt_logs;
        throw f;
    }
}

template <typename SparseMatrixType, typename TestMatrixType>
void CheckAllConfigurations(const TestMatrixType& test_matrix,
                            const std::string& arg_name,
                            const std::string& filename = "unknown",
                            int lineno = -1)
{
    using ValueType = typename SparseMatrixType::value_type;
    using Format = typename SparseMatrixType::format;
    using DeviceTestMatrix = typename TestMatrixType::rebind<cusp::device_memory>::type;

    // prepare the x vector
    cusp::array1d<ValueType, cusp::host_memory> host_x(test_matrix.num_cols);
    for(size_t i = 0; i < host_x.size(); i++) {
        host_x[i] = i % 10;
    }
    cusp::array1d<ValueType, cusp::device_memory> device_x = host_x;
    cusp::array1d<ValueType, cusp::host_memory> reference_y(test_matrix.num_rows, 10);

    const DeviceTestMatrix* device_test_matrix = nullptr;

    DeviceTestMatrix device_matrix;
    if constexpr (std::is_same_v<cusp::host_memory, typename TestMatrixType::memory_space>) {
        device_matrix = test_matrix;
        device_test_matrix = &device_matrix;
    } else {
        device_test_matrix = &test_matrix;
    }

    // Compute the reference output on the gpu.
    // Do in in a nested block so that all memory is deallocated in the end
    // and we don't take up device memory.
    {
        cusp::array1d<ValueType, cusp::device_memory> y(test_matrix.num_rows, 10);

        cusp::ktt::disable();
        cusp::multiply(*device_test_matrix, device_x, y);
        cusp::ktt::enable();

        reference_y = y;
    }

    SparseMatrixType A = *device_test_matrix;
    cusp::array1d<ValueType, cusp::host_memory> host_y(A.num_rows, 10);
    cusp::array1d<ValueType, cusp::device_memory> device_y = host_y;

    auto& tuner = cusp::ktt::get_tuner();

    std::stringstream logging_stream;
    tuner.SetLoggingTarget(logging_stream);

    ::ktt::ReferenceComputation reference_computation = [&] (void* raw_buffer) {
        std::memcpy(raw_buffer, (void*) reference_y.data(), sizeof(ValueType)*reference_y.size());
    };

    auto results = cusp::ktt::tune(A, device_x, device_y, reference_computation);

    tuner.SetLoggingTarget(std::cerr);
    std::string logs = logging_stream.str();

    assert_tunning_results_valid(results, logs, arg_name, filename, lineno);
}

#define CHECK_ALL_CONFIGURATIONS(MatrixTypeUnderTest, input_matrix) \
    CheckAllConfigurations<MatrixTypeUnderTest>(input_matrix, #input_matrix, __FILE__, __LINE__)


template <class TestMatrix>
void TestSparseMatrixVectorMultiply()
{
    using ValueType = typename TestMatrix::value_type;
    using IndexType = typename TestMatrix::index_type;

    cusp::array2d<ValueType, cusp::host_memory> A(5,4);
    A(0, 0) = 13;
    A(0, 1) = 80;
    A(0, 2) =  0;
    A(0, 3) =  0;
    A(1, 0) =  0;
    A(1, 1) = 27;
    A(1, 2) =  0;
    A(1, 3) =  0;
    A(2, 0) = 55;
    A(2, 1) =  0;
    A(2, 2) = 24;
    A(2, 3) = 42;
    A(3, 0) =  0;
    A(3, 1) = 69;
    A(3, 2) =  0;
    A(3, 3) = 83;
    A(4, 0) =  0;
    A(4, 1) =  0;
    A(4, 2) = 27;
    A(4, 3) =  0;
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> A_coo = A;
    CHECK_ALL_CONFIGURATIONS(TestMatrix, A_coo);

    cusp::array2d<ValueType,cusp::host_memory> B(2,4);
    B(0,0) = 0.0;
    B(0,1) = 2.0;
    B(0,2) = 3.0;
    B(0,3) = 4.0;
    B(1,0) = 5.0;
    B(1,1) = 0.0;
    B(1,2) = 0.0;
    B(1,3) = 8.0;
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> B_coo = B;
    CHECK_ALL_CONFIGURATIONS(TestMatrix, B_coo);

    cusp::array2d<ValueType,cusp::host_memory> C(2,2);
    C(0,0) = 0.0;
    C(0,1) = 0.0;
    C(1,0) = 3.0;
    C(1,1) = 5.0;
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> C_coo = C;
    CHECK_ALL_CONFIGURATIONS(TestMatrix, C_coo);

    cusp::array2d<ValueType,cusp::host_memory> D(2,1);
    D(0,0) = 2.0;
    D(1,0) = 3.0;
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> D_coo = D;
    CHECK_ALL_CONFIGURATIONS(TestMatrix, D_coo);

    cusp::array2d<ValueType,cusp::host_memory> F(2,3);
    F(0,0) = 0.0;
    F(0,1) = 1.5;
    F(0,2) = 3.0;
    F(1,0) = 0.5;
    F(1,1) = 0.0;
    F(1,2) = 0.0;
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> F_coo = F;
    CHECK_ALL_CONFIGURATIONS(TestMatrix, F_coo);

    auto G_dia = cusp::ktt::make_diagonal_symmetric_matrix(4096, 4096, 1, 1024);
    CHECK_ALL_CONFIGURATIONS(TestMatrix, G_dia);

    auto H_dia = cusp::ktt::make_diagonal_symmetric_matrix(4096, 2048, 1, 1024);
    CHECK_ALL_CONFIGURATIONS(TestMatrix, H_dia);

    auto I_dia = cusp::ktt::make_diagonal_symmetric_matrix(2048, 4096, 1, 1024);
    CHECK_ALL_CONFIGURATIONS(TestMatrix, I_dia);
}
DECLARE_KTT_UNITTEST(TestSparseMatrixVectorMultiply);
