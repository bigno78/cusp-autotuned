#pragma once

#include <cusp/ktt/detail/ktt.inl>

#define STR(str) #str
#define STRING(str) STR(str)

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

template<typename T>
void* cast(const T* ptr) { 
    return const_cast<void*>(static_cast<const void*>(ptr));
}

template<typename Array>
::ktt::ArgumentId add_arg(const Array& array, ::ktt::ArgumentAccessType access = ::ktt::ArgumentAccessType::ReadOnly) {
    auto& tuner = *cusp::ktt::detail::tuner;
    auto id = tuner.AddArgumentVector(cast(array.data().get()),
                                    array.size(),
                                    sizeof(decltype(array.front())),
                                    access,
                                    ::ktt::ArgumentMemoryLocation::Device);
    if (id == ::ktt::InvalidArgumentId) {
        std::cerr << "ERROR: Adding argument failed and I don't know why\n";
    }
    return id;
}

} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
