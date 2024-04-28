#pragma once

#include <Ktt.h>

#include <memory>
#include <cstdlib>
#include <utility>

#include <cuda.h>

#include <cusp/system/cuda/ktt/multiply.h>

namespace cusp {

namespace ktt {

namespace detail {


inline std::unique_ptr<::ktt::Tuner> tuner;
inline bool is_enabled = true;


inline void cleanup()
{
    tuner.reset();
}

inline void lazy_init()
{
    if (is_enabled && !tuner) {
        cuInit(0);

        CUdevice device;
        cuDeviceGet(&device, 0);

        CUcontext context;
        cuCtxGetCurrent(&context);

        if (context == nullptr) {
            tuner = std::make_unique<::ktt::Tuner>(0, 0, ::ktt::ComputeApi::CUDA);
        } else {
            CUstream stream;
            cuStreamCreate(&stream, CU_STREAM_DEFAULT);

            std::vector<::ktt::ComputeQueue> queues = { stream };
            ::ktt::ComputeApiInitializer initializer(context, queues);

            tuner = std::make_unique<::ktt::Tuner>(::ktt::ComputeApi::CUDA,
                                                   initializer);
        }

        std::string compiler_flags = "-std=c++17 ";
#ifdef KTT_LINE_INFO
        compiler_flags += "-lineinfo ";
#endif
        tuner->SetCompilerOptions(compiler_flags);
        tuner->SetValidationMode(::ktt::ValidationMode::OfflineTuning);

        std::atexit(cleanup);
    }
}


} // namespace detail


inline void disable() {
    detail::is_enabled = false;
}

inline void enable() {
    detail::is_enabled = true;
}

inline ::ktt::Tuner& get_tuner()
{
    detail::lazy_init();
    return *detail::tuner;
}


template <template<typename, typename, typename> typename Matrix,
          typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
::ktt::KernelResult multiply(
     const Matrix<IndexType, ValueType1, cusp::device_memory>& A,
     const cusp::array1d<ValueType2, cusp::device_memory>& x,
     cusp::array1d<ValueType3, cusp::device_memory>& y)
{
    return cusp::system::cuda::ktt::multiply(get_tuner(), A, x, y);
}


template <template<typename, typename, typename> typename Matrix,
          typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
::ktt::KernelResult multiply(
     const Matrix<IndexType, ValueType1, cusp::device_memory>& A,
     const cusp::array1d<ValueType2, cusp::device_memory>& x,
     cusp::array1d<ValueType3, cusp::device_memory>& y,
     const ::ktt::KernelConfiguration& configuration,
     bool run_with_profiling)
{
    return cusp::system::cuda::ktt::multiply(get_tuner(), A, x, y, configuration, run_with_profiling);
}


template <template<typename, typename, typename> typename Matrix,
          typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
std::vector<::ktt::KernelResult>
tune(const Matrix<IndexType, ValueType1, cusp::device_memory>& A,
     const cusp::array1d<ValueType2, cusp::device_memory>& x,
     cusp::array1d<ValueType3, cusp::device_memory>& y,
     std::optional<::ktt::ReferenceComputation> reference_computation,
     std::unique_ptr<::ktt::StopCondition> stop_condition,
     std::unique_ptr<::ktt::Searcher> searcher)
{
    return cusp::system::cuda::ktt::tune(get_tuner(), A, x, y, reference_computation,
                                         std::move(stop_condition), std::move(searcher));
}

template <typename MatrixType,
          typename ValueType1,
          typename ValueType2,
          typename MemorySpace1,
          typename MemorySpace2>
void reset_tuning(const MatrixType& A,
                  const cusp::array1d<ValueType1, MemorySpace1>& x,
                  cusp::array1d<ValueType2, MemorySpace2>& y)
{
    auto& tuner = get_tuner();
    const auto& kernel = cusp::system::cuda::ktt::get_kernel(tuner, A, x, y);
    tuner.ClearData(kernel.kernel_id);
}


} // namespace ktt

} // namespace cusp
