#pragma once

#include <Ktt.h>
#include <cusp/system/cuda/ktt/kernel.h>

#include <cstdint>   // uint64_t
#include <stdexcept> // std::runtime_error


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
::ktt::ArgumentId add_arg(::ktt::Tuner& tuner, T scalar)
{
    auto id = tuner.AddArgumentScalar(scalar);
    if (id == ::ktt::InvalidArgumentId) {
        std::cerr << "ERROR: Adding scalar argument failed and I don't know why\n";
    }
    return id;
}

template<typename T>
::ktt::ArgumentId add_arg(::ktt::Tuner& tuner,
             const cusp::array1d<T, cusp::device_memory>& array)
{
    auto id = tuner.AddArgumentVector<T>(cast(array.data().get()),
                                         array.size() * sizeof(T),
                                         ::ktt::ArgumentAccessType::ReadOnly,
                                         ::ktt::ArgumentMemoryLocation::Device);
    if (id == ::ktt::InvalidArgumentId) {
        std::cerr << "ERROR: Adding const array argument failed and I don't know why\n";
    }
    return id;
}

template<typename T>
::ktt::ArgumentId add_arg(::ktt::Tuner& tuner,
             cusp::array1d<T, cusp::device_memory>& array)
{
    auto id = tuner.AddArgumentVector<T>(cast(array.data().get()),
                                         array.size() * sizeof(T),
                                         ::ktt::ArgumentAccessType::ReadWrite,
                                         ::ktt::ArgumentMemoryLocation::Device);
    if (id == ::ktt::InvalidArgumentId) {
        std::cerr << "ERROR: Adding non-const array argument failed and I don't know why\n";
    }
    return id;
}

template<typename T>
void add_arg(::ktt::Tuner& tuner, std::vector<::ktt::ArgumentId>& argument_ids, T&& arg)
{
    auto id = add_arg(tuner, std::forward<T>(arg));
    argument_ids.push_back(id);
}

template<typename... Args>
std::vector<::ktt::ArgumentId> add_arguments(::ktt::Tuner& tuner, Args&&... args)
{
    std::vector<::ktt::ArgumentId> argument_ids;
    ( add_arg(tuner, argument_ids, std::forward<Args>(args)), ... );
    return argument_ids;
}

inline void remove_arguments(const kernel_context& kernel, const std::vector<::ktt::ArgumentId>& args)
{
    for (auto id : kernel.definition_ids) {
        kernel.tuner->SetArguments(id, {});
    }

    for (auto arg : args) {
        kernel.tuner->RemoveArgument(arg);
    }
}


uint64_t get_parameter_uint(const ::ktt::KernelConfiguration& conf,
                            const std::string& name)
{
    for (const auto& pair : conf.GetPairs())
        if (pair.GetName() == name)
            return pair.GetValue();

    throw std::runtime_error("No paramater with name: " + name);
}

double get_parameter_double(const ::ktt::KernelConfiguration& conf,
                            const std::string& name)
{
    for (const auto& pair : conf.GetPairs())
        if (pair.GetName() == name)
            return pair.GetValueDouble();

    throw std::runtime_error("No paramater with name: " + name);
}

} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
