#pragma once

#include <cusp/detail/config.h>

#include <cusp/ktt/detail/external/nameof.hpp>
#include <cusp/system/cuda/ktt/utils.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/system/cuda/utils.h>

#include <thrust/fill.h>

#include <iostream>


namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

namespace csr {


inline void setup_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    tuner.AddParameter(kernel_id, "BLOCK_SIZE", std::vector<uint64_t>{ 128 });
    tuner.AddParameter(kernel_id, "THREADS_PER_VECTOR", std::vector<uint64_t>{ 32 });

    tuner.AddThreadModifier(
        kernel.kernel_id,
        { kernel.definition_ids[0] },
        ::ktt::ModifierType::Local,
        ::ktt::ModifierDimension::X,
        { std::string("BLOCK_SIZE") },
        [] (const uint64_t defaultSize, const std::vector<uint64_t>& parameters) {
            return parameters[0];
        }
    );
}


template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         typename ValueType3>
kernel_context initialize_kernel(::ktt::Tuner& tuner)
{
    std::string kernel_path =
        STRING(CUSP_PATH) "/cusp/system/cuda/ktt/kernels/csr_kernel.h";

    kernel_context kernel(tuner);

    std::vector< std::string > type_names {
        std::string(NAMEOF_TYPE(IndexType)),
        std::string(NAMEOF_TYPE(ValueType1)),
        std::string(NAMEOF_TYPE(ValueType2)),
        std::string(NAMEOF_TYPE(ValueType3)),
    };

    // NOTE: These can be anything since they are awlays set in the launcher.
    // So use some values that will hopefully cause a crash if not set properly
    // in the launcher.
    ::ktt::DimensionVector block_size(0);
    ::ktt::DimensionVector grid_size(0);

    auto definition_id = tuner.AddKernelDefinitionFromFile(
        "ktt_csr_vector_kernel",
        kernel_path,
        grid_size,
        block_size,
        type_names
    );

    kernel.definition_ids.push_back(definition_id);
    kernel.kernel_id = tuner.CreateSimpleKernel("CsrKernel", definition_id);

    setup_tuning_parameters(kernel);

    return kernel;
}


} // namespace csr


template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         typename ValueType3>
const kernel_context& get_kernel(::ktt::Tuner& tuner, cusp::csr_format format)
{
    static kernel_context kernel =
        csr::initialize_kernel<IndexType, ValueType1, ValueType2, ValueType3>(tuner);

    return kernel;
}


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
std::vector<::ktt::ArgumentId>
add_arguments(const kernel_context& kernel,
              const cusp::csr_matrix<IndexType, ValueType1, cusp::device_memory>& A,
              const cusp::array1d<ValueType2, cusp::device_memory>& x,
              cusp::array1d<ValueType3, cusp::device_memory>& y)
{
    auto args = add_arguments(*kernel.tuner,
                              A.num_rows, A.row_offsets, A.column_indices,
                              A.values, x, y);

    kernel.tuner->SetArguments(kernel.definition_ids[0], args);

    return args;
}


// Returns the argument id of the y vector given a vector of arguments
// returned by a previous call of `add_arguments`.
inline ::ktt::ArgumentId
get_output_argument(const std::vector<::ktt::ArgumentId>& arguments,
                    cusp::csr_format)
{
    return arguments[5];
}


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
auto get_launcher(const kernel_context& ctx,
                  const cusp::csr_matrix<IndexType, ValueType1, cusp::device_memory>& A,
                  const cusp::array1d<ValueType2, cusp::device_memory>& x,
                  cusp::array1d<ValueType3, cusp::device_memory>& y,
                  bool profile = false)
{
    return [&] (::ktt::ComputeInterface& interface)
    {
        auto conf = interface.GetCurrentConfiguration();

        ::ktt::DimensionVector block_size =
            interface.GetCurrentLocalSize(ctx.definition_ids[0]);

        auto threads_per_vector = get_parameter_uint(conf, "THREADS_PER_VECTOR");
        auto vectors_per_block = block_size.GetSizeX() / threads_per_vector;

        ::ktt::DimensionVector grid_size(
            DIVIDE_INTO(A.num_rows, vectors_per_block));

        if (!profile) {
            interface.RunKernel(ctx.definition_ids[0], grid_size, block_size);
        } else {
#ifdef PROFILE
            interface.RunKernelWithProfiling(ctx.definition_ids[0],
                                             grid_size, block_size);
#endif
        }
    };
}


} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
