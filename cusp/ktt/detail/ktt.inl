#pragma once

#include <Ktt.h>

#include <memory>
#include <cstdlib>

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

            ::ktt::ComputeApiInitializer initializer(context, std::vector<::ktt::ComputeQueue>{ stream });

            tuner = std::make_unique<::ktt::Tuner>(::ktt::ComputeApi::CUDA, initializer);
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


template <typename Matrix,
          typename ValueType1,
          typename ValueType2>
::ktt::KernelResult multiply(const Matrix& A,
              const cusp::array1d<ValueType1, cusp::device_memory>& x,
              cusp::array1d<ValueType2, cusp::device_memory>& y)
{
    return cusp::system::cuda::ktt::multiply(get_tuner(), A, x, y);
}


template <typename Matrix,
          typename ValueType1,
          typename ValueType2>
::ktt::KernelResult multiply(const Matrix& A,
                             const cusp::array1d<ValueType1, cusp::device_memory>& x,
                             cusp::array1d<ValueType2, cusp::device_memory>& y,
                             const ::ktt::KernelConfiguration& configuration,
                             bool run_with_profiling)
{
    return cusp::system::cuda::ktt::multiply(get_tuner(), A, x, y, configuration, run_with_profiling);
}


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
std::vector<::ktt::KernelResult>
tune(const cusp::dia_matrix<IndexType, ValueType1, cusp::device_memory>& A,
     const cusp::array1d<ValueType2, cusp::device_memory>& x,
     cusp::array1d<ValueType3, cusp::device_memory>& y,
     std::optional<::ktt::ReferenceComputation> reference_computation)
{
    return cusp::system::cuda::ktt::tune(get_tuner(), A, x, y, reference_computation);
}


template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         typename ValueType3,
         typename Format>
void reset_tuning()
{
    using namespace cusp::system::cuda::ktt;

    kernel_context kernel = get_kernel<IndexType, ValueType1, ValueType2, ValueType3>(get_tuner(), Format{});
    detail::tuner->ClearData(kernel.kernel_id);
}


template <typename MatrixType,
          typename ValueType1,
          typename ValueType2>
void reset_tuning(const MatrixType& A,
                  const cusp::array1d<ValueType1, cusp::device_memory>& x,
                  cusp::array1d<ValueType2, cusp::device_memory>& y)
{
    using IndexType = typename MatrixType::index_type;
    using ValueType = typename MatrixType::value_type;
    using Format = typename MatrixType::format;

    return reset_tuning<IndexType, ValueType, ValueType2, ValueType2, Format>();
}


} // namespace ktt

} // namespace cusp