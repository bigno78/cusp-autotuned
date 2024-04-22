#pragma once

#include <cusp/detail/config.h>

#include <cusp/system/cuda/ktt/utils.h>
#include <cusp/system/cuda/ktt/common.h>

#include <cusp/coo_matrix.h>
#include <cusp/array1d.h>
#include <cusp/system/cuda/utils.h>

#include <thrust/fill.h>

#include <iostream>
#include <utility>      // pair


namespace cusp::system::cuda::ktt {

namespace coo {

inline bool MORE_VALUES_PER_THREAD = false;

inline void setup_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    tuner.AddParameter(kernel_id, "BLOCK_SIZE", u64_vec{ 128, 256, 512 });
    tuner.AddParameter(kernel_id, "VALUES_PER_THREAD", u64_vec{ 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 15, 17, 19, 20, 21, 33 });
    tuner.AddParameter(kernel_id, "IMPL", u64_vec{ 0, 1, 2, 3 });

    tuner.AddParameter(kernel_id, "USE_CARRY", u64_vec{ 0, 1 });
    tuner.AddParameter(kernel_id, "AVOID_ATOMIC", u64_vec{ 0, 1 });

    tuner.AddConstraint(kernel_id, { "USE_CARRY", "IMPL" },
        [](const auto& vals)
        {
            if (vals[0] == 1) return vals[1] == 0;
            return true;
        });

    tuner.AddConstraint(kernel_id, { "AVOID_ATOMIC", "IMPL", "VALUES_PER_THREAD" },
        [](const auto& vals)
        {
            if (vals[0] == 1) return (vals[1] == 1 && vals[2] != 1)
                                   || vals[1] == 2 || vals[1] == 3;
            return true;
        });

    // tuner.AddConstraint(kernel_id, { "VALUES_PER_THREAD", "IMPL" },
    //     [](const std::vector<uint64_t>& vals)
    //     {
    //         if (vals[0] == 1 && vals[1] == 2) return false;
    //         return true;
    //     });

    tuner.AddThreadModifier(kernel.kernel_id,
            { kernel.definition_ids[0] },
            ::ktt::ModifierType::Local,
            ::ktt::ModifierDimension::X,
            { std::string("BLOCK_SIZE") },
            [](const uint64_t default_size, const u64_vec& parameters) {
                return parameters[0];
            });
    tuner.AddThreadModifier(kernel.kernel_id,
            { kernel.definition_ids[1] },
            ::ktt::ModifierType::Local,
            ::ktt::ModifierDimension::X,
            { std::string("BLOCK_SIZE") },
            [](const uint64_t default_size, const u64_vec& parameters) {
                return parameters[0];
            });
}


template<typename Idx, typename Val1, typename Val2, typename Val3>
kernel_context initialize_kernel(::ktt::Tuner& tuner)
{
    kernel_context kernel(tuner);

    const ::ktt::DimensionVector blockDimensions(0);
    const ::ktt::DimensionVector gridDimensions(0);

    // auto definition_id = tuner.AddKernelDefinitionFromFile(
    //     "coo_spmv",
    //     KernelsPath + "coo_kernel.h",
    //     gridDimensions,
    //     blockDimensions,
    //     names_of_types<Idx, Val1, Val2, Val3>()
    // );
    // kernel.definition_ids.push_back(definition_id);
    // kernel.kernel_id = tuner.CreateSimpleKernel("CooKernel", definition_id);

    auto def_zero = tuner.AddKernelDefinitionFromFile(
        "zero_output",
        KernelsPath + "coo_kernel.h",
        gridDimensions,
        blockDimensions,
        names_of_types<Idx, Val1, Val2, Val3>()
    );
    auto def_spmv = tuner.AddKernelDefinitionFromFile(
        "coo_spmv",
        KernelsPath + "coo_kernel.h",
        gridDimensions,
        blockDimensions,
        names_of_types<Idx, Val1, Val2, Val3>()
    );
    kernel.definition_ids.push_back(def_zero);
    kernel.definition_ids.push_back(def_spmv);
    kernel.kernel_id = tuner.CreateCompositeKernel("CooKernel",
                                                   { def_zero, def_spmv });

    coo::setup_tuning_parameters(kernel);

    return kernel;
}



struct grid_config
{
    int block_count = 0;
    int block_count_zeroing = 0;
};

inline grid_config get_grid_config(size_t input_size, int vals_per_thread, int block_size)
{
    grid_config result{};
    result.block_count         = DIVIDE_INTO(input_size, vals_per_thread * block_size);
    result.block_count_zeroing = DIVIDE_INTO(input_size, block_size);
    return result;
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
    kernel.tuner->SetArguments(kernel.definition_ids[1], args);

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

        using DimVec = ::ktt::DimensionVector;

        DimVec block_size = interface.GetCurrentLocalSize(ctx.definition_ids[0]);
        auto block_count = DIVIDE_INTO(A.num_entries, vals_per_thread * block_size.GetSizeX());
        DimVec grid_size(block_count);
        DimVec zero_grid_size(DIVIDE_INTO(A.num_entries, block_size.GetSizeX()));

        if (!profile) {
            interface.RunKernel(ctx.definition_ids[0], zero_grid_size, block_size);
            interface.RunKernel(ctx.definition_ids[1], grid_size, block_size);
        } else {
            interface.RunKernelWithProfiling(ctx.definition_ids[0], zero_grid_size, block_size);
            interface.RunKernelWithProfiling(ctx.definition_ids[1], grid_size, block_size);
        }
    };
}


} // namespace cusp::system::cuda::ktt
