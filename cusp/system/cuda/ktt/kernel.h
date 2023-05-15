#pragma once

#include <Ktt.h>

#include <vector>

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {


struct kernel_context {
    ::ktt::Tuner* tuner = nullptr;
    ::ktt::KernelId kernel_id = ::ktt::InvalidKernelId;

    // This is a vector to allow for composite kernels.
    std::vector<::ktt::KernelDefinitionId> definition_ids;

    kernel_context(::ktt::Tuner& tuner) : tuner(&tuner) { }

    kernel_context(::ktt::Tuner& tuner,
                   const std::vector<::ktt::KernelDefinitionId>& definition_ids,
                   ::ktt::KernelId kernel_id)
        : tuner(&tuner),
          definition_ids(definition_ids),
          kernel_id(kernel_id) { }
};


} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
