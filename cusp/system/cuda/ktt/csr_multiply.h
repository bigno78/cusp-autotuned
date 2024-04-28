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


inline int* row_starts = nullptr;
inline int max_workers = 0;
inline float last_row_starts_compute_us = -1;

inline int* row_counter = nullptr;


inline void reset_row_counter()
{
    cudaMemset(row_counter, 0, sizeof(int));
}


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

        while (count <= w * chunk && w * chunk < next && w < workers)
        {
            out[w] = i;
            ++w;
        }

        count = next;
    }
    for (; w < workers; ++w)
        out[w] = 0;
}


template<typename Idx>
__global__
void gpu_compute_row_starts(const unsigned int num_rows,
                            const Idx* __restrict__ Ar,
                            const unsigned int num_entries,
                            Idx* __restrict__ row_starts,
                            const int workers, const int chunk)
{
    const int ti = blockDim.x * blockIdx.x + threadIdx.x;

    if (ti >= num_rows)
        return;

    int count = Ar[ ti ];
    int next  = Ar[ ti + 1 ];

    for (int w = count / chunk; w < workers && w * chunk < next; ++w)
    {
        if (count <= w * chunk)
            row_starts[w] = ti;
    }
}


template<typename Mat, typename Vec>
inline void device_compute_row_starts(const Mat& A, Vec* out, int workers)
{
    int chunk = DIVIDE_INTO(A.num_entries, workers);

    int block_size = 256;
    int block_count = DIVIDE_INTO(A.num_rows, block_size);

    // Important to reset the vector, because the kernel might not
    // assign workers that fall outside of bounds. Since those
    // workers would get no work anyway, it's okay to set the vector
    // to large values before the computation.
    cudaMemset(out, 0x7f, workers);

    gpu_compute_row_starts<<<block_count, block_size>>>(
                                A.num_rows, A.row_offsets.data().get(),
                                A.num_entries, out, workers, chunk);
}



template<typename Mat>
void update_row_starts(int block_count, int block_size, int threads_per_row,
                       const Mat& A)
{
    if (threads_per_row == 0) threads_per_row = 32;

    int warps_in_block = block_size / threads_per_row;
    int workers = block_count * warps_in_block;

    float delta_ms = 0;
    cudaEvent_t cu_start, cu_stop;
    cudaEventCreate(&cu_start);
    cudaEventCreate(&cu_stop);
    cudaEventRecord(cu_start);

    device_compute_row_starts(A, row_starts, workers);

    cudaEventRecord(cu_stop);
    cudaEventSynchronize(cu_stop);
    cudaEventElapsedTime(&delta_ms, cu_start, cu_stop);

    last_row_starts_compute_us = delta_ms * 1000;
}




inline void setup_tuning_parameters(const kernel_context& kernel)
{
    auto& tuner = *kernel.tuner;
    auto kernel_id = kernel.kernel_id;

    tuner.AddParameter(kernel_id, "BLOCK_SIZE", std::vector<uint64_t>{ 128, 256, 512 });

    auto dev_info = tuner.GetCurrentDeviceInfo();
    auto u = dev_info.GetMaxComputeUnits();
    tuner.AddParameter(kernel_id, "BLOCKS", std::vector<uint64_t>{ u / 2,
                                                                   u,
                                                                   u * 2,
                                                                   u * 4,
                                                                   u * 8,
                                                                   u * 16 });

    tuner.AddParameter(kernel_id, "THREADS_PER_ROW", std::vector<uint64_t>{ 0, 1, 2, 4, 8, 16, 32 });

    tuner.AddParameter(kernel_id, "DYNAMIC", std::vector<uint64_t>{ 0, 1, 2 });

    tuner.AddParameter(kernel_id, "AVOID_ATOMIC", std::vector<uint64_t>{ 0, 1 });
    tuner.AddParameter(kernel_id, "ALIGNED", std::vector<uint64_t>{ 0, 1 });
    tuner.AddParameter(kernel_id, "SPECIAL_LOADS", std::vector<uint64_t>{ 0, 1, 2, 3 });

    tuner.AddParameter(kernel_id, "UNROLL", std::vector<uint64_t>{ 0, 1, 2, 4, 8, 16, 32 });

    tuner.AddConstraint(kernel_id, { "ALIGNED", "THREADS_PER_ROW" },
        [](const std::vector<uint64_t>& vals)
        {
            if (vals[0] == 1) return vals[1] == 32 || vals[1] == 0;
            return true;
        });

    tuner.AddConstraint(kernel_id, { "AVOID_ATOMIC", "DYNAMIC" },
        [](const std::vector<uint64_t>& vals)
        {
            if (vals[0] == 1) return vals[1] == 2;
            return true;
        });

    tuner.AddConstraint(kernel_id, { "THREADS_PER_ROW", "DYNAMIC" },
        [](const std::vector<uint64_t>& vals)
        {
            if (vals[1] == 2) return vals[0] == 32
                                  || vals[0] == 16
                                  || vals[0] == 8
                                  || vals[0] == 4
                                  || vals[0] == 2
                                  || vals[0] == 1;
            return true;
        });

    max_workers = (u * 32) * 512;

    if (row_starts)
        cudaFree(row_starts);

    cudaMalloc(&row_starts,    max_workers * sizeof(int));
    cudaMemset(row_starts,  0, max_workers * sizeof(int));

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

        auto block_count = get_parameter_uint(conf, "BLOCKS");
        ::ktt::DimensionVector grid_size(block_count);

        auto dynamic = get_parameter_uint(conf, "DYNAMIC");

        if (dynamic == 2)
            csr::update_row_starts(block_count,
                                   get_parameter_uint(conf, "BLOCK_SIZE"),
                                   get_parameter_uint(conf, "THREADS_PER_ROW"),
                                   A);
        else
            csr::last_row_starts_compute_us = -1;

        if (dynamic == 1)
            csr::reset_row_counter();

        if (!profile) {
            interface.RunKernel(ctx.definition_ids[0], grid_size, block_size);
        } else {
            interface.RunKernelWithProfiling(ctx.definition_ids[0],
                                             grid_size, block_size);
        }
    };
}


} // namespace cusp::system::cuda::ktt
