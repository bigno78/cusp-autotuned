#pragma once 

#include <iostream>

#include <cusp/ktt/detail/external/nameof.hpp>
#include <cusp/system/cuda/arch.h> // max_active_blocks

#include <cusp/system/cuda/ktt/kernels/dia_kernel.h>
#include <cusp/system/cuda/ktt/kernel.h>
#include <cusp/system/cuda/ktt/utils.h>


namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

namespace dia {

inline void setup_tuning_parameters(::ktt::Tuner& tuner, const kernel_context& kernel)
{
    tuner.AddParameter(kernel.kernel_id, "TEST_PARAM", std::vector<uint64_t>{0, 1, 2});
}

} // namespace dia


template<typename IndexType, typename ValueType1, typename ValueType2, typename ValueType3>
kernel_context initialize_kernel(::ktt::Tuner& tuner, cusp::dia_format)
{
    kernel_context kernel(tuner);

    std::string kernel_path = STRING(CUSP_PATH) "/cusp/system/cuda/ktt/kernels/dia_kernel.h";

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(
                                ktt_dia_vector_kernel<IndexType, ValueType1, ValueType2, ValueType3, BLOCK_SIZE>,
                                BLOCK_SIZE, 0);

    std::vector< std::string > type_names {
        std::string(NAMEOF_TYPE(IndexType)),
        std::string(NAMEOF_TYPE(ValueType1)),
        std::string(NAMEOF_TYPE(ValueType2)),
        std::string(NAMEOF_TYPE(ValueType3)),
        std::to_string(BLOCK_SIZE)
    };

    const ::ktt::DimensionVector blockDimensions(BLOCK_SIZE);
    const ::ktt::DimensionVector gridDimensions(MAX_BLOCKS);

    kernel.definition_id = tuner.AddKernelDefinitionFromFile(
        "ktt_dia_vector_kernel", 
        kernel_path, 
        gridDimensions,
        blockDimensions,
        type_names
    );
    kernel.kernel_id = tuner.CreateSimpleKernel("DiaKernel", kernel.definition_id);
    
    dia::setup_tuning_parameters(tuner, kernel);

    return kernel;
}

template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         typename ValueType3>
kernel_context get_kernel(::ktt::Tuner& tuner, cusp::dia_format format)
{
    static kernel_context kernel = initialize_kernel<IndexType, ValueType1, ValueType2, ValueType3>(tuner, format);
    return kernel;
}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
std::vector<::ktt::ArgumentId> add_arguments(
                                    const kernel_context& kernel,
                                    const cusp::dia_matrix<IndexType, ValueType1, cusp::device_memory>& A,
                                    const cusp::array1d<ValueType2, cusp::device_memory>& x,
                                    cusp::array1d<ValueType3, cusp::device_memory>& y,
                                    cusp::dia_format format)
{
    const IndexType num_diagonals = A.values.num_cols;
    const IndexType pitch = A.values.pitch; // length of one diagonal + possible padding

    auto args = add_arguments(*kernel.tuner, A.num_rows, A.num_cols, num_diagonals, pitch, A.diagonal_offsets, A.values.values, x, y);
    kernel.tuner->SetArguments(kernel.definition_id, args); 

    return args;
}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
void multiply(::ktt::Tuner& tuner,
              const cusp::dia_matrix<IndexType, ValueType1, cusp::device_memory>& A,
              const cusp::array1d<ValueType2, cusp::device_memory>& x,
              cusp::array1d<ValueType3, cusp::device_memory>& y,
              cusp::dia_format format)
{
    if (A.num_entries == 0) {
        thrust::fill(y.begin(), y.end(), ValueType3(0));
        return;
    }
    
    kernel_context kernel = get_kernel<IndexType, ValueType1, ValueType2, ValueType3>(tuner, format);
    auto args = add_arguments(kernel, A, x, y, format);
    tuner.TuneIteration(kernel.kernel_id, {});
    remove_arguments(kernel, args);
}

} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
