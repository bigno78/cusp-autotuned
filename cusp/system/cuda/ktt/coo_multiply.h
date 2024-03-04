#pragma once

#include <cusp/detail/config.h>

#include <cusp/system/cuda/ktt/utils.h>
#include <cusp/system/cuda/ktt/common.h>

#include <cusp/coo_matrix.h>
#include <cusp/array1d.h>
#include <cusp/system/cuda/utils.h>

#include <thrust/fill.h>

#include <iostream>


namespace cusp::system::cuda::ktt {

namespace coo {

inline void setup_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    tuner.AddParameter(kernel_id, "VALUES_PER_THREAD", u64_vec{ 1, 2, 4, 8, 16, 32, 64 });
    tuner.AddParameter(kernel_id, "BLOCK_SIZE", u64_vec{ 128, 256, 512 });

    tuner.AddThreadModifier(kernel.kernel_id,
            { kernel.definition_ids[0] },
            ::ktt::ModifierType::Local,
            ::ktt::ModifierDimension::X,
            { std::string("BLOCK_SIZE") },
            [](const uint64_t defaultSize, const u64_vec& parameters) {
                return parameters[0];
            });
}


template<typename Idx, typename Val1, typename Val2, typename Val3>
kernel_context initialize_kernel(::ktt::Tuner& tuner)
{
    kernel_context kernel(tuner);

    const ::ktt::DimensionVector blockDimensions(0);
    const ::ktt::DimensionVector gridDimensions(0);

    auto definition_id = tuner.AddKernelDefinitionFromFile(
        "coo_spmv",
        KernelsPath + "coo_kernel.h",
        gridDimensions,
        blockDimensions,
        names_of_types<Idx, Val1, Val2, Val3>()
    );
    kernel.definition_ids.push_back(definition_id);
    kernel.kernel_id = tuner.CreateSimpleKernel("CooKernel", definition_id);

    coo::setup_tuning_parameters(kernel);

    return kernel;
}


} // namespace coo


template<typename Idx, typename Val1,typename Val2, typename Val3,
         typename MemorySpace>
const kernel_context& get_kernel(::ktt::Tuner& tuner,
                const cusp::coo_matrix<Idx, Val1, MemorySpace>&)
{
    static kernel_context kernel =
        coo::initialize_kernel<Idx, Val1, Val2, Val3>(tuner);

    return kernel;
}


template <typename Idx, typename Val1, typename Val2, typename Val3>
auto add_arguments(const kernel_context& kernel,
                   const cusp::coo_matrix<Idx, Val1, cusp::device_memory>& A,
                   const cusp::array1d<Val2, cusp::device_memory>& x,
                         cusp::array1d<Val3, cusp::device_memory>& y)
    -> std::vector<::ktt::ArgumentId>
{
    auto args =
        add_arguments(*kernel.tuner, A.row_indices, A.column_indices, A.values,
                      A.values.size(), x, y, y.size());

    kernel.tuner->SetArguments(kernel.definition_ids[0], args);

    return args;
}


inline auto get_output_argument(const std::vector<::ktt::ArgumentId>& arguments,
                                cusp::coo_format)
    -> ::ktt::ArgumentId
{
    return arguments[5];
}


template <typename Idx, typename Val1, typename Val2, typename Val3>
auto get_launcher(const kernel_context& ctx,
                  const cusp::coo_matrix<Idx, Val1, cusp::device_memory>& A,
                  const cusp::array1d<Val2, cusp::device_memory>& x,
                  cusp::array1d<Val3, cusp::device_memory>& y,
                  bool profile = false)
{
    return [&, profile](::ktt::ComputeInterface& interface)
    {
        const auto& conf = interface.GetCurrentConfiguration();
        auto vals_per_thread = get_parameter_uint(conf, "VALUES_PER_THREAD");

        ::ktt::DimensionVector block_size =
            interface.GetCurrentLocalSize(ctx.definition_ids[0]);
        ::ktt::DimensionVector grid_size(
            DIVIDE_INTO(A.num_entries, vals_per_thread * block_size.GetSizeX()));

        if (!profile) {
            interface.RunKernel(ctx.definition_ids[0], grid_size, block_size);
        } else {
            interface.RunKernelWithProfiling(ctx.definition_ids[0],
                                             grid_size, block_size);
        }
    };
}


} // namespace cusp::system::cuda::ktt
