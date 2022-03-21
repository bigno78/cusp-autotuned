#pragma once

#include <Ktt.h>

#include <memory>

#include <cuda.h>

namespace cusp {

namespace ktt {

namespace detail {

std::unique_ptr<::ktt::Tuner> tuner;
bool is_enabled = true;

void lazy_init() {

    if (is_enabled && !tuner) {
        CUdevice device;
        cuDeviceGet(&device, 0);

        CUcontext context;
        cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);

        CUstream stream;
        cuStreamCreate(&stream, CU_STREAM_DEFAULT);

        ::ktt::ComputeApiInitializer initializer(context, std::vector<::ktt::ComputeQueue>{stream});

        tuner = std::make_unique<::ktt::Tuner>(::ktt::ComputeApi::CUDA, initializer);
    }
}

} // namespace detail

} // namespace ktt

} // namespace cusp
