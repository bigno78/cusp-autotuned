#pragma once

#include <cusp/detail/config.h>

#include <cusp/ktt/detail/external/nameof.hpp>
#include <cusp/system/cuda/ktt/utils.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/system/cuda/utils.h>

#include <thrust/fill.h>

#include <iostream>


namespace cusp::system::cuda::ktt {

namespace csr {


inline void setup_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    tuner.AddParameter(kernel_id, "ROWS_PER_BLOCK", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64, 128 });
    // tuner.AddParameter(kernel_id, "ROWS_PER_BLOCK", std::vector<uint64_t>{ 8 });
    // tuner.AddParameter(kernel_id, "ROWS_PER_BLOCK", std::vector<uint64_t>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
    // tuner.AddParameter(kernel_id, "ROWS_PER_BLOCK", std::vector<uint64_t>{ 1 });

    tuner.AddParameter(kernel_id, "BLOCK_SIZE", std::vector<uint64_t>{ 32, 64, 128, 256, 512 });
    // tuner.AddParameter(kernel_id, "BLOCK_SIZE", std::vector<uint64_t>{ 64 });
    // tuner.AddParameter(kernel_id, "BLOCK_SIZE", std::vector<uint64_t>{ 128 });

    tuner.AddParameter(kernel_id, "THREADS_PER_ROW", std::vector<uint64_t>{ 0, 1, 2, 4, 8, 16, 32 });
    // tuner.AddParameter(kernel_id, "THREADS_PER_ROW", std::vector<uint64_t>{ 32 });
    tuner.AddParameter(kernel_id, "DYNAMIC", std::vector<uint64_t>{ 0, 1 });

    tuner.AddConstraint(kernel_id, { "THREADS_PER_ROW", "DYNAMIC" },
        [](const std::vector<uint64_t>& vals)
        {
            // return vals[1] != 1 || vals[0] == 32;
            if (vals[1] == 1) return vals[0] == 32;
            return true;
        });

    tuner.AddThreadModifier(
        kernel.kernel_id,
        { kernel.definition_ids[0] },
        ::ktt::ModifierType::Local,
        ::ktt::ModifierDimension::X,
        { std::string("BLOCK_SIZE") },
        [](const uint64_t defaultSize, const std::vector<uint64_t>& parameters) {
            return parameters[0];
        }
    );
}


int* row_counter = nullptr;


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
        "csr_spmv",
        kernel_path,
        grid_size,
        block_size,
        type_names
    );

    kernel.definition_ids.push_back(definition_id);
    kernel.kernel_id = tuner.CreateSimpleKernel("CsrKernel", definition_id);

    setup_tuning_parameters(kernel);

    if (row_counter)
        cudaFree(row_counter);

    cudaMalloc(&row_counter, sizeof(int));
    cudaMemset(row_counter, 0, sizeof(int));
    printf("cudaMalloc, add_arguments\n");

    return kernel;
}


} // namespace csr


template<typename Idx, typename Val1,typename Val2, typename Val3,
         typename MemorySpace>
const kernel_context& get_kernel(::ktt::Tuner& tuner,
                const cusp::csr_matrix<Idx, Val1, MemorySpace>&)
{
    static kernel_context kernel =
        csr::initialize_kernel<Idx, Val1, Val2, Val3>(tuner);

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
                              A.values, x, y,
                              csr::row_counter);

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
    return [&, profile](::ktt::ComputeInterface& interface)
    {
        auto conf = interface.GetCurrentConfiguration();

        ::ktt::DimensionVector block_size =
            interface.GetCurrentLocalSize(ctx.definition_ids[0]);

        auto rows_per_block = get_parameter_uint(conf, "ROWS_PER_BLOCK");
        auto threads_per_row = get_parameter_uint(conf, "THREADS_PER_ROW");

        unsigned block_count = 0;

        if (threads_per_row == 0)
            block_count = DIVIDE_INTO( A.num_rows, rows_per_block );
        else
            block_count = DIVIDE_INTO( A.num_rows, rows_per_block * block_size.GetSizeX() / threads_per_row );

        ::ktt::DimensionVector grid_size(block_count);

        cudaMemset(csr::row_counter, 0, sizeof(int));

        if (!profile) {
            interface.RunKernel(ctx.definition_ids[0], grid_size, block_size);
        } else {
            interface.RunKernelWithProfiling(ctx.definition_ids[0],
                                             grid_size, block_size);
        }
    };
}


} // namespace cusp::system::cuda::ktt
