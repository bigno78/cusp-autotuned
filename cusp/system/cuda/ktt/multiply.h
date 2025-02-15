#pragma once

#include <cusp/system/cuda/ktt/kernel.h>

#include <cusp/system/cuda/ktt/coo_multiply.h>
#include <cusp/system/cuda/ktt/csr_multiply.h>
#include <cusp/system/cuda/ktt/dia_multiply.h>
#include <cusp/system/cuda/ktt/ell_multiply.h>

#include <Ktt.h>

#include <optional>
#include <memory>
#include <utility>

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {


// Helpful function to get a kernel used for a given multiplication
// instead of having to manually specify the types.
template <typename MatrixType,
          typename ValueType1,
          typename ValueType2,
          typename MemorySpace1,
          typename MemorySpace2>
kernel_context get_kernel(
    ::ktt::Tuner& tuner,
    const MatrixType& A,
    const cusp::array1d<ValueType1, MemorySpace1>& x,
    cusp::array1d<ValueType2, MemorySpace2>& y)
{
    using IndexType = typename MatrixType::index_type;
    using ValueType = typename MatrixType::value_type;

    if constexpr (std::is_same_v<cusp::host_memory,
                                 typename MatrixType::memory_space>)
    {
        using DeviceMatrix =
            typename MatrixType::rebind<cusp::device_memory>::type;
        return get_kernel<IndexType, ValueType, ValueType1, ValueType2>(
                    tuner, DeviceMatrix{});
    }
    else
    {
        return get_kernel<IndexType, ValueType, ValueType1, ValueType2>(tuner, A);
    }
}


template <typename MatrixType,
          typename VectorType1,
          typename VectorType2>
::ktt::KernelResult
multiply(::ktt::Tuner& tuner,
         const MatrixType& A,
         const VectorType1& x,
         VectorType2& y)
{
    const kernel_context& kernel = get_kernel(tuner, A, x, y);

    auto args = add_arguments(kernel, A, x, y);

    auto launcher = get_launcher(kernel, A, x, y, false);
    tuner.SetLauncher(kernel.kernel_id, launcher);

    auto res = tuner.TuneIteration(kernel.kernel_id, {});

    remove_arguments(kernel, args);

    return res;
}


template <typename MatrixType,
          typename VectorType1,
          typename VectorType2>
::ktt::KernelResult
multiply(::ktt::Tuner& tuner,
         const MatrixType& A,
         const VectorType1& x,
         VectorType2& y,
         const ::ktt::KernelConfiguration& configuration,
         bool run_with_profiling = false)
{
    const kernel_context& kernel = get_kernel(tuner, A, x, y);

    auto args = add_arguments(kernel, A, x, y);

    auto launcher = get_launcher(kernel, A, x, y, run_with_profiling);
    tuner.SetLauncher(kernel.kernel_id, launcher);

    auto result = tuner.Run(kernel.kernel_id, configuration, {});

    remove_arguments(kernel, args);

    return result;
}


template <typename MatrixType,
          typename VectorType1,
          typename VectorType2>
std::vector<::ktt::KernelResult>
tune(::ktt::Tuner& tuner,
     const MatrixType& A,
     const VectorType1& x,
     VectorType2& y,
     std::optional<::ktt::ReferenceComputation> ref_computation = std::nullopt,
     std::unique_ptr<::ktt::StopCondition> stop_condition = nullptr,
     std::unique_ptr<::ktt::Searcher> searcher = nullptr)
{
    using Format = typename MatrixType::format;
    using HostVector2 =
        typename VectorType2::template rebind<cusp::host_memory>::type;

    kernel_context kernel = get_kernel(tuner, A, x, y);
    auto args = add_arguments(kernel, A, x, y);

    if (ref_computation)
    {
        tuner.SetReferenceComputation(get_output_argument(args, Format{}),
                                      *ref_computation);
    }

    if (searcher)
        tuner.SetSearcher(kernel.kernel_id, std::move(searcher));

    HostVector2 host_y = y;
    tuner.SetLauncher(kernel.kernel_id, [&] (::ktt::ComputeInterface& interface)
    {
        // clear y so previous results don't affect validation
        y = host_y;
        auto launcher = get_launcher(kernel, A, x, y, false);
        launcher(interface);
    });

    auto results = stop_condition
            ? tuner.Tune(kernel.kernel_id, std::move(stop_condition))
            : tuner.Tune(kernel.kernel_id);

    remove_arguments(kernel, args);

    if (searcher)
        tuner.SetSearcher(kernel.kernel_id, std::make_unique<::ktt::DeterministicSearcher>());

    return results;
}


} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
