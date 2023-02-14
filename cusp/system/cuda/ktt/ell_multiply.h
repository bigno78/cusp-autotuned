#pragma once

#include <cusp/system/cuda/ktt/kernel.h>
#include <cusp/system/cuda/ktt/utils.h>

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

namespace ell {


inline void setup_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    tuner.AddParameter(kernel_id, "BLOCK_SIZE", std::vector<uint64_t>{ 128 });
    tuner.AddParameter(kernel_id, "UNCACHED_LOADS",
                        std::vector<uint64_t>{ 0, 1 });

    tuner.AddThreadModifier(
        kernel_id,
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
        STRING(CUSP_PATH) "/cusp/system/cuda/ktt/kernels/ell_kernel.h";

    kernel_context kernel(tuner);

    std::vector< std::string > type_names {
        std::string(NAMEOF_TYPE(IndexType)),
        std::string(NAMEOF_TYPE(ValueType1))
    };

    // NOTE: These can be anything since they are awlays set in the launcher.
    // So use some values that will hopefully cause a crash if not set properly
    // in the launcher.
    ::ktt::DimensionVector block_size(0);
    ::ktt::DimensionVector grid_size(0);

    auto definition_id = tuner.AddKernelDefinitionFromFile(
        "ktt_ell_kernel",
        kernel_path,
        grid_size,
        block_size,
        type_names
    );

    kernel.definition_ids.push_back(definition_id);
    kernel.kernel_id = tuner.CreateSimpleKernel("EllKernel", definition_id);

    setup_tuning_parameters(kernel);

    return kernel;
}


} // namespace ell


template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         typename ValueType3>
const kernel_context& get_kernel(::ktt::Tuner& tuner, cusp::ell_format format)
{
    static kernel_context kernel =
        ell::initialize_kernel<IndexType, ValueType1, ValueType2, ValueType3>(tuner);

    return kernel;
}


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
std::vector<::ktt::ArgumentId>
add_arguments(const kernel_context& kernel,
              const cusp::ell_matrix<IndexType, ValueType1, cusp::device_memory>& A,
              const cusp::array1d<ValueType2, cusp::device_memory>& x,
              cusp::array1d<ValueType3, cusp::device_memory>& y)
{
    IndexType pitch = A.column_indices.pitch;
    IndexType num_entries_per_row = A.column_indices.num_cols;

    auto args = add_arguments(*kernel.tuner,
                              A.num_rows, A.num_cols, num_entries_per_row, pitch,
                              A.column_indices.values, A.values.values, x, y);

    kernel.tuner->SetArguments(kernel.definition_ids[0], args);

    return args;
}


// Returns the argument id of the y vector given a vector of arguments
// returned by a previous call of `add_arguments`.
inline ::ktt::ArgumentId
get_output_argument(const std::vector<::ktt::ArgumentId>& arguments,
                    cusp::ell_format)
{
    return arguments[7];
}


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
auto get_launcher(const kernel_context& ctx,
                  const cusp::ell_matrix<IndexType, ValueType1, cusp::device_memory>& A,
                  const cusp::array1d<ValueType2, cusp::device_memory>& x,
                  cusp::array1d<ValueType3, cusp::device_memory>& y,
                  bool profile = false)
{
    return [&] (::ktt::ComputeInterface& interface)
    {
        ::ktt::DimensionVector block_size =
            interface.GetCurrentLocalSize(ctx.definition_ids[0]);
        ::ktt::DimensionVector grid_size(
            DIVIDE_INTO(A.num_rows, block_size.GetSizeX()));

        if (!profile) {
            interface.RunKernel(ctx.definition_ids[0], grid_size, block_size);
        } else {
            interface.RunKernelWithProfiling(ctx.definition_ids[0],
                                             grid_size, block_size);
        }
    };
}


} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
