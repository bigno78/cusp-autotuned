#pragma once

#include <cusp/detail/config.h>

#include <cusp/system/cuda/ktt/kernel.h>
#include <cusp/system/cuda/ktt/utils.h>
#include <cusp/ktt/detail/external/nameof.hpp>

#include <cusp/system/cuda/arch.h> // max_active_blocks
#include <cusp/system/cuda/utils.h>
#include <cusp/dia_matrix.h>

#include <optional>
#include <iostream>

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

namespace dia {

inline void setup_tuning_parameters(::ktt::Tuner& tuner, const kernel_context& kernel)
{
    tuner.AddParameter(kernel.kernel_id, "KERNEL_TYPE", std::vector<uint64_t>{ 0, 1, 2 });
    tuner.AddParameter(kernel.kernel_id, "SHARED_PREFETCH_FACTOR", std::vector<uint64_t>{ 0, 2, 4, 8 });
    tuner.AddParameter(kernel.kernel_id, "REGISTER_PREFETCH_FACTOR", std::vector<uint64_t>{ 0, 2, 3, 4 });
    tuner.AddParameter(kernel.kernel_id, "REGISTER_PREFETCH_TYPE", std::vector<uint64_t>{ 0, 1 });
    tuner.AddParameter(kernel.kernel_id, "LOAD_TYPE", std::vector<uint64_t>{ 0, 1 });
    tuner.AddParameter(kernel.kernel_id, "STRIPING_FACTOR", std::vector<uint64_t>{ 2, 4, 8 });
    tuner.AddParameter(kernel.kernel_id, "BLOCK_SIZE", std::vector<uint64_t>{ 256, 512 });

    // START OF HACKS ---------------------
    tuner.AddParameter(kernel.kernel_id, "BLOCKS_PER_SECTOR", std::vector<uint64_t>{ 0 });
    tuner.AddParameter(kernel.kernel_id, "CHUNK_SIZE", std::vector<uint64_t>{ 0 });
    tuner.AddParameter(kernel.kernel_id, "STRIPE_SIZE", std::vector<uint64_t>{ 0 });
    tuner.AddParameter(kernel.kernel_id, "GRID_SIZE", std::vector<uint64_t>{ 0 });
    tuner.AddParameter(kernel.kernel_id, "SECTOR_MAPPING_TYPE",
                       std::vector<uint64_t>{ 0, 1 });

    tuner.AddConstraint(kernel.kernel_id, { "SECTOR_MAPPING_TYPE" },
                        [] (const std::vector<uint64_t>& values) {
                            return values[0] == 0;
                        });
    // END OF HACKS -----------------------

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

    // only one type of prefetching can be applied at once
    tuner.AddConstraint(kernel.kernel_id,
                        { "SHARED_PREFETCH_FACTOR", "REGISTER_PREFETCH_FACTOR" },
                        [] (const std::vector<uint64_t>& values) {
                            return values[0] == 0 || values[1] == 0;
                        });

    // prefetching is used only for the blocked offsets kernel
    tuner.AddConstraint(kernel.kernel_id,
                        { "KERNEL_TYPE", "SHARED_PREFETCH_FACTOR", "REGISTER_PREFETCH_FACTOR" },
                        [] (const std::vector<uint64_t>& values) {
                            return values[0] == 1 || (values[1] == 0 && values[2] == 0);
                        });

    // different register prefetching implementations are used only when register prefetching is enabled
    tuner.AddConstraint(kernel.kernel_id,
                        { "REGISTER_PREFETCH_FACTOR", "REGISTER_PREFETCH_TYPE" },
                        [] (const std::vector<uint64_t>& values) {
                            return values[0] > 0 || values[1] == 0;
                        });

    // don't try different striping factors for non-striped kernels
    tuner.AddConstraint(kernel.kernel_id,
                    { "KERNEL_TYPE", "STRIPING_FACTOR" },
                    [] (const std::vector<uint64_t>& values) {
                        return values[0] == 2 || values[1] == 2;
                    });
}

template<typename IndexType, typename ValueType1, typename ValueType2, typename ValueType3>
kernel_context initialize_kernel(::ktt::Tuner& tuner)
{
    kernel_context kernel(tuner);

    std::string kernel_path = STRING(CUSP_PATH) "/cusp/system/cuda/ktt/kernels/dia_kernel.h";

    std::vector< std::string > type_names {
        std::string(NAMEOF_TYPE(IndexType)),
        std::string(NAMEOF_TYPE(ValueType1)),
        std::string(NAMEOF_TYPE(ValueType2)),
        std::string(NAMEOF_TYPE(ValueType3))
    };

    const ::ktt::DimensionVector blockDimensions(0);
    const ::ktt::DimensionVector gridDimensions(0);

    auto definition_id = tuner.AddKernelDefinitionFromFile(
        "ktt_dia_vector_kernel",
        kernel_path,
        gridDimensions,
        blockDimensions,
        type_names
    );
    kernel.definition_ids.push_back(definition_id);
    kernel.kernel_id = tuner.CreateSimpleKernel("DiaKernel", definition_id);

    dia::setup_tuning_parameters(tuner, kernel);

    return kernel;
}


} // namespace dia


template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         typename ValueType3,
         typename MemorySpace>
const kernel_context& get_kernel(
        ::ktt::Tuner& tuner,
        const cusp::dia_matrix<IndexType, ValueType1, MemorySpace>& A)
{
    static kernel_context kernel = dia::initialize_kernel<IndexType, ValueType1, ValueType2, ValueType3>(tuner);
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
                                    cusp::array1d<ValueType3, cusp::device_memory>& y)
{
    const IndexType num_diagonals = A.values.num_cols;
    const IndexType pitch = A.values.pitch; // length of one diagonal + possible padding

    auto args = add_arguments(*kernel.tuner, A.num_rows, A.num_cols, num_diagonals, pitch, A.diagonal_offsets, A.values.values, x, y);
    kernel.tuner->SetArguments(kernel.definition_ids[0], args);

    return args;
}


// Returns the argument id of the y vector given a vector of arguments returned by a previous
// call of `add_arguments`.
inline ::ktt::ArgumentId get_output_argument(const std::vector<::ktt::ArgumentId>& arguments, cusp::dia_format) {
    return arguments[7];
}


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
auto get_launcher(
        const kernel_context& ctx,
        const cusp::dia_matrix<IndexType, ValueType1, cusp::device_memory>& A,
        const cusp::array1d<ValueType2, cusp::device_memory>& x,
        cusp::array1d<ValueType3, cusp::device_memory>& y,
        bool profile = false)
{
    return [&, profile] (::ktt::ComputeInterface& interface)
    {
        ::ktt::DimensionVector block_size = interface.GetCurrentLocalSize(ctx.definition_ids[0]);
        ::ktt::DimensionVector grid_size( DIVIDE_INTO(A.num_rows, block_size.GetSizeX()) );

        auto conf = interface.GetCurrentConfiguration();
        for (const auto& pair : conf.GetPairs())
            if (pair.GetName() == "GRID_SIZE" && pair.GetValue() != 0)
                grid_size = ::ktt::DimensionVector(pair.GetValue());

        if (!profile) {
            interface.RunKernel(ctx.definition_ids[0], grid_size, block_size);
        } else {
            interface.RunKernelWithProfiling(ctx.definition_ids[0], grid_size, block_size);
        }
    };
}


} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
