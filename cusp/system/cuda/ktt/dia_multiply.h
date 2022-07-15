#pragma once 

#include <iostream>

#include <cusp/ktt/detail/external/nameof.hpp>
#include <cusp/ktt/detail/ktt.inl>
#include <cusp/system/cuda/arch.h> // max_active_blocks

#include <cusp/system/cuda/ktt/kernels/dia_kernel.h>

namespace cusp::system::cuda::ktt::dia {
    inline ::ktt::KernelId get_kernel_id(::ktt::KernelDefinitionId id,  std::string kernel_name) {
        static ::ktt::KernelId kernel = cusp::ktt::detail::tuner->CreateSimpleKernel(kernel_name, id);
        return kernel;
    }
}

template<>
inline ::ktt::KernelId cusp::ktt::detail::get_kernel_id<cusp::dia_format>() {
    return cusp::system::cuda::ktt::dia::get_kernel_id(0, ""); 
}

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

namespace dia {

inline void setup_tuning_parameters(::ktt::Tuner& tuner, ::ktt::KernelId kernel) {
    tuner.AddParameter(kernel, "TEST_PARAM", std::vector<uint64_t>{0, 1, 2});
}

} // dia

template <typename DerivedPolicy,
          typename MatrixType,
          typename VectorType1,
          typename VectorType2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              const MatrixType& A,
              const VectorType1& x,
              VectorType2& y,
              cusp::dia_format)
{
    static_assert( std::is_same_v<cusp::device_memory, typename MatrixType::memory_space>
                   && std::is_same_v<cusp::device_memory, typename VectorType1::memory_space>
                   && std::is_same_v<cusp::device_memory, typename VectorType2::memory_space>, 
                   "All arguments must be in device memory." );

    using ValueType = typename MatrixType::value_type;

    if (A.num_entries == 0) {
        thrust::fill(y.begin(), y.end(), ValueType(0));
        return;
    }

    std::cout << "Hello from dia\n";

    using IndexType = typename MatrixType::index_type;

    auto& tuner = *cusp::ktt::detail::tuner;

    std::string path = STRING(CUSP_PATH) "/cusp/system/cuda/ktt/kernels/dia_kernel.h";

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(
                               ktt_dia_vector_kernel<typename MatrixType::index_type, typename MatrixType::value_type,
                                  typename VectorType1::value_type, typename VectorType2::value_type, BLOCK_SIZE>,
                               BLOCK_SIZE, (size_t) sizeof(IndexType) * BLOCK_SIZE);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, BLOCK_SIZE));

    const IndexType num_diagonals = A.values.num_cols;
    const IndexType pitch         = A.values.pitch;
    
    std::vector< std::string > type_names {
        std::string(NAMEOF_TYPE(typename MatrixType::index_type)),
        std::string(NAMEOF_TYPE(typename MatrixType::value_type)),
        std::string(NAMEOF_TYPE(typename VectorType1::value_type)),
        std::string(NAMEOF_TYPE(typename VectorType2::value_type)),
        std::to_string(BLOCK_SIZE)
    };

    const ::ktt::DimensionVector blockDimensions(BLOCK_SIZE);
    const ::ktt::DimensionVector gridDimensions(NUM_BLOCKS);

    ::ktt::KernelDefinitionId definition = tuner.GetKernelDefinitionId("ktt_dia_vector_kernel", type_names);

    bool called_first_time = definition == ::ktt::InvalidKernelDefinitionId; 

    if (called_first_time) {
        definition = tuner.AddKernelDefinitionFromFile(
            "ktt_dia_vector_kernel", 
            path, 
            gridDimensions,
            blockDimensions,
            type_names);
    }

    ::ktt::KernelId kernel = dia::get_kernel_id(definition, "DiaKernel");

    if (called_first_time) {
        dia::setup_tuning_parameters(tuner, kernel);
    }

    auto num_rows_id = tuner.AddArgumentScalar(A.num_rows);
    auto num_cols_id = tuner.AddArgumentScalar(A.num_cols);
    auto num_diagonals_id = tuner.AddArgumentScalar(num_diagonals);
    auto pitch_id = tuner.AddArgumentScalar(pitch);
    auto diagonal_offsets_id = add_arg(A.diagonal_offsets);
    auto vals_id = add_arg(A.values.values);
    auto x_id = add_arg(x);
    auto y_id = add_arg(y, ::ktt::ArgumentAccessType::ReadWrite);

    std::vector<::ktt::ArgumentId> args = { 
        num_rows_id, num_cols_id, num_diagonals_id, pitch_id, diagonal_offsets_id, vals_id, x_id, y_id 
    };

    tuner.SetArguments(definition, args);

    tuner.TuneIteration(kernel, {});

    tuner.SetArguments(definition, {});
    for (auto arg : args) {
        tuner.RemoveArgument(arg);
    }
}

} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
