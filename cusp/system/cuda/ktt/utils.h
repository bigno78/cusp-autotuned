#pragma once

#include <Ktt.h>
#include <cusp/system/cuda/ktt/kernel.h>

#define STR(str) #str
#define STRING(str) STR(str)

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

template<typename T>
void* cast(const T* ptr)
{ 
    return const_cast<void*>(static_cast<const void*>(ptr));
}

template<typename T>
void add_arg(::ktt::Tuner& tuner, std::vector<::ktt::ArgumentId>& argument_ids, T scalar)
{
    auto id = tuner.AddArgumentScalar(scalar);
    if (id == ::ktt::InvalidArgumentId) {
        std::cerr << "ERROR: Adding scalar argument failed and I don't know why\n";
    }
    argument_ids.push_back(id);
}

template<typename T>
void add_arg(::ktt::Tuner& tuner,
             std::vector<::ktt::ArgumentId>& argument_ids,
             const cusp::array1d<T, cusp::device_memory>& array)
{
    auto id = tuner.AddArgumentVector<T>(cast(array.data().get()),
                                         array.size(),
                                         ::ktt::ArgumentAccessType::ReadOnly,
                                         ::ktt::ArgumentMemoryLocation::Device);
    if (id == ::ktt::InvalidArgumentId) {
        std::cerr << "ERROR: Adding const array argument failed and I don't know why\n";
    }
    argument_ids.push_back(id);
}

template<typename T>
void add_arg(::ktt::Tuner& tuner,
             std::vector<::ktt::ArgumentId>& argument_ids,
             cusp::array1d<T, cusp::device_memory>& array)
{
    auto id = tuner.AddArgumentVector<T>(cast(array.data().get()),
                                         array.size(),
                                         ::ktt::ArgumentAccessType::ReadWrite,
                                         ::ktt::ArgumentMemoryLocation::Device);
    if (id == ::ktt::InvalidArgumentId) {
        std::cerr << "ERROR: Adding non-const array argument failed and I don't know why\n";
    }
    argument_ids.push_back(id);
}

template<typename... Args> 
std::vector<::ktt::ArgumentId> add_arguments(::ktt::Tuner& tuner, Args&&... args)
{
    std::vector<::ktt::ArgumentId> argument_ids;
    ( add_arg(tuner, argument_ids, args), ... );
    return argument_ids;
}

void remove_arguments(const kernel_context& kernel, const std::vector<::ktt::ArgumentId>& args)
{
    kernel.tuner->SetArguments(kernel.definition_id, {});
    for (auto arg : args) {
        kernel.tuner->RemoveArgument(arg);
    }   
}


} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
