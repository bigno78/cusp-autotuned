#pragma once

#include <Ktt.h>

#include <memory>
#include <cstdlib>

#include <cuda.h>


namespace cusp {

namespace ktt {

namespace detail {

inline std::unique_ptr<::ktt::Tuner> tuner;
inline bool is_enabled = true;


template<typename Format>
::ktt::KernelId get_kernel_id();


inline void cleanup() {
    tuner.reset();
}

inline void lazy_init() {
    if (is_enabled && !tuner) {
        CUdevice device;
        cuDeviceGet(&device, 0);

        CUcontext context;
        cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);

        CUstream stream;
        cuStreamCreate(&stream, CU_STREAM_DEFAULT);

        ::ktt::ComputeApiInitializer initializer(context, std::vector<::ktt::ComputeQueue>{stream});

        tuner = std::make_unique<::ktt::Tuner>(::ktt::ComputeApi::CUDA, initializer);

        std::atexit(cleanup);
    }
}


} // namespace detail

template<typename Matrix>
void reset_tuning(const Matrix& matrix) {
    auto id = detail::get_kernel_id<typename Matrix::format>();
    detail::tuner->ClearData(id);
}

} // namespace ktt

} // namespace cusp
