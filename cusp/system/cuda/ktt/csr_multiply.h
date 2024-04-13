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


// inline int* row_starts = nullptr;
inline cusp::array1d<int, cusp::device_memory> row_starts;


inline int* row_counter = nullptr;


template<typename Mat, typename Vec>
inline void cpu_compute_row_starts(const Mat& A, Vec& out, int workers)
{
    int count = 0;
    int w = 0;

    int total = A.num_entries;
    int chunk = DIVIDE_INTO(total, workers);

    for (int i = 0; i < A.row_offsets.size() - 1; ++i)
    {
        int next = A.row_offsets[ i + 1 ];

        while (count <= w * chunk && w * chunk < next)
        {
            out[w] = i;
            ++w;
        }

        count = next;
    }
    // TODO: isn't there a corner case when not all workers have been assigned?
}


inline void setup_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    // tuner.AddParameter(kernel_id, "ROWS_PER_WORKER", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64, 128 });
    // tuner.AddParameter(kernel_id, "ROWS_PER_WORKER", std::vector<uint64_t>{ 1, 4, 64, 128, 10'000, 100'000 });

    tuner.AddParameter(kernel_id, "BLOCK_SIZE", std::vector<uint64_t>{ 32, 64, 128, 256, 512 });

    auto dev_info = tuner.GetCurrentDeviceInfo();
    auto u = dev_info.GetMaxComputeUnits();
    tuner.AddParameter(kernel_id, "NUMBER_OF_BLOCKS", std::vector<uint64_t>{ u / 2, u, u * 2, u * 4, u * 8, u * 16 });

    tuner.AddParameter(kernel_id, "THREADS_PER_ROW", std::vector<uint64_t>{ 0, 1, 2, 4, 8, 16, 32 });

    // tuner.AddParameter(kernel_id, "DYNAMIC", std::vector<uint64_t>{ 0, 1 });
    tuner.AddParameter(kernel_id, "DYNAMIC", std::vector<uint64_t>{ 2 });

    // tuner.AddConstraint(kernel_id, { "THREADS_PER_ROW", "DYNAMIC" },
    //     [](const std::vector<uint64_t>& vals)
    //     {
    //         if (vals[1] == 1) return vals[0] == 32
    //                               || vals[0] == 1
    //                               || vals[0] == 0;
    //         return true;
    //     });

    unsigned max_workers = (u * 16) * 512;
    row_starts = decltype(row_starts){ max_workers };

    // if (row_starts) cudaFree(row_starts);
    // cudaMalloc(&row_starts,   max_workers * sizeof(int));
    // cudaMemset(row_starts, 0, max_workers * sizeof(int));

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
                              A.num_entries,
                              csr::row_counter,
                              csr::row_starts);

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

        // auto rows_per_worker = get_parameter_uint(conf, "ROWS_PER_WORKER");
        // auto threads_per_row = get_parameter_uint(conf, "THREADS_PER_ROW");

        // unsigned block_count = 0;

        // if (threads_per_row == 0)
        //     block_count = DIVIDE_INTO( A.num_rows, rows_per_worker );
        // else
        //     block_count = DIVIDE_INTO( A.num_rows, rows_per_worker * block_size.GetSizeX() / threads_per_row );

        // ::ktt::DimensionVector grid_size(block_count);

        auto block_count = get_parameter_uint(conf, "NUMBER_OF_BLOCKS");
        ::ktt::DimensionVector grid_size(block_count);

        if (get_parameter_uint(conf, "DYNAMIC") == 2)
        {
            struct fake_mat
            {
                unsigned long num_entries = 0;
                cusp::array1d<IndexType, cusp::host_memory> row_offsets;
            };

            namespace krn = std::chrono;
            auto start = krn::steady_clock::now();

            auto local_row_starts = cusp::array1d<int, cusp::host_memory>{ csr::row_starts.size() };

            auto fake = fake_mat{ A.num_entries, A.row_offsets };

            auto warps_in_block = get_parameter_uint(conf, "BLOCK_SIZE") / 32;
            csr::cpu_compute_row_starts(fake, local_row_starts, block_count * warps_in_block);

            csr::row_starts = local_row_starts;

            auto end = krn::steady_clock::now();
            auto time = krn::duration_cast<krn::microseconds>(end - start).count();
            std::cout << "csr::cpu_compute_row_starts: " << time << " us" << std::endl;
        }

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
