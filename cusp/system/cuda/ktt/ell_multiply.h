#pragma once

#include <cusp/ktt/ellr_matrix.h>

#include <cusp/system/cuda/ktt/kernel.h>
#include <cusp/system/cuda/ktt/utils.h>
#include <cusp/ell_matrix.h>

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

namespace ell {


inline void setup_common_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    tuner.AddParameter(kernel_id, "BLOCK_SIZE",      u64_vec{ 128, 256, 512 });
    tuner.AddParameter(kernel_id, "UNCACHED_LOADS",  u64_vec{ 0, 1 });
    tuner.AddParameter(kernel_id, "DISABLE_UNROLL",  u64_vec{ 0, 1 });
    tuner.AddParameter(kernel_id, "PREFETCH_FACTOR", u64_vec{ 0, 2, 4 });
    tuner.AddParameter(kernel_id, "THREADS_PER_ROW", u64_vec{ 1, 2, 4 });

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

    // All the warps working on one stripe of the matrix fit in a thread block.
    tuner.AddConstraint(
            kernel.kernel_id,
            { "THREADS_PER_ROW", "BLOCK_SIZE" },
            [] (const std::vector<uint64_t>& values) {
                return 32*values[0] <= values[1];
            });
}

inline void setup_ell_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    setup_common_tuning_parameters(kernel);

    tuner.AddParameter(kernel_id, "BREAK", u64_vec{ 0, 1 });

    // BREAK can be used only when there is no prefetching
    // TODO(KTT): Implement prefetching when using early break and remove this
    tuner.AddConstraint(
            kernel.kernel_id,
            { "BREAK", "PREFETCH_FACTOR" },
            [] (const std::vector<uint64_t>& values) {
                return values[0] == 0 || values[1] == 0;
            });
}

inline void setup_ellr_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    setup_common_tuning_parameters(kernel);

    tuner.AddParameter(kernel_id, "ELLR", u64_vec{ 1 });
}


template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         typename ValueType3,
         typename TuningParamsInit>
kernel_context initialize_kernel(::ktt::Tuner& tuner,
                                 const std::string& kernel_function,
                                 const std::string& kernel_name,
                                 TuningParamsInit init)
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
        kernel_function,
        kernel_path,
        grid_size,
        block_size,
        type_names
    );

    kernel.definition_ids.push_back(definition_id);
    kernel.kernel_id = tuner.CreateSimpleKernel(kernel_name, definition_id);

    init(kernel);

    return kernel;
}


} // namespace ell


template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         typename ValueType3>
const kernel_context& get_kernel(
        ::ktt::Tuner& tuner,
        const cusp::ell_matrix<IndexType, ValueType1, cusp::device_memory>& A)
{
    static kernel_context kernel =
        ell::initialize_kernel<IndexType, ValueType1, ValueType2, ValueType3>(
            tuner, "ktt_ell_kernel", "EllKernel",
            ell::setup_ell_tuning_parameters);

    return kernel;
}

template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         typename ValueType3>
const kernel_context& get_kernel(
    ::ktt::Tuner& tuner,
    const cusp::ktt::ellr_matrix<IndexType, ValueType1, cusp::device_memory>& A)
{
    static kernel_context kernel =
        ell::initialize_kernel<IndexType, ValueType1, ValueType2, ValueType3>(
            tuner, "ktt_ellr_kernel", "EllrKernel",
            ell::setup_ellr_tuning_parameters);

    return kernel;
}


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
std::vector<::ktt::ArgumentId>
add_common_arguments(
        const kernel_context& kernel,
        const cusp::ell_matrix<IndexType, ValueType1, cusp::device_memory>& A,
        const cusp::array1d<ValueType2, cusp::device_memory>& x,
        cusp::array1d<ValueType3, cusp::device_memory>& y)
{
    IndexType pitch = A.column_indices.pitch;
    IndexType num_entries_per_row = A.column_indices.num_cols;

    return add_arguments(*kernel.tuner, A.num_rows, A.num_cols,
                         num_entries_per_row, pitch, A.column_indices.values,
                         A.values.values, x, y);
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
    auto args = add_common_arguments(kernel, A, x, y);
    kernel.tuner->SetArguments(kernel.definition_ids[0], args);
    return args;
}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
std::vector<::ktt::ArgumentId>
add_arguments(
    const kernel_context& kernel,
    const cusp::ktt::ellr_matrix<IndexType, ValueType1, cusp::device_memory>& A,
    const cusp::array1d<ValueType2, cusp::device_memory>& x,
    cusp::array1d<ValueType3, cusp::device_memory>& y)
{
    auto args = add_common_arguments(kernel, A, x, y);
    auto aditional_args = add_arguments(*kernel.tuner, A.row_lengths);

    args.push_back(aditional_args[0]);

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
        const auto& conf = interface.GetCurrentConfiguration();
        auto threads_per_row = get_parameter_uint(conf, "THREADS_PER_ROW");

        auto threads_in_block =
            interface.GetCurrentLocalSize(ctx.definition_ids[0]).GetSizeX();

        ::ktt::DimensionVector block_size(
            threads_in_block/threads_per_row, threads_per_row);
        ::ktt::DimensionVector grid_size(
            DIVIDE_INTO(A.num_rows, threads_in_block/threads_per_row));

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
