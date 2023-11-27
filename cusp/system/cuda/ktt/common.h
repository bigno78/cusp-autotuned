#pragma once

#include <cusp/system/cuda/ktt/utils.h>
#include <cusp/ktt/detail/external/nameof.hpp>

#include <string>
#include <vector>

inline std::string KernelsPath = std::string(STRING(CUSP_PATH))
                               + "/cusp/system/cuda/ktt/kernels/";


template<typename T, typename ... Args>
inline void push(std::vector<std::string>& out)
{
    out.push_back(std::string(NAMEOF_TYPE(T)));

    if constexpr (sizeof...(Args) > 0)
        push<Args...>(out);
}


template<typename ... Args>
inline auto names_of_types() -> std::vector<std::string>
{
    auto vec = std::vector<std::string>{};
    vec.reserve(sizeof...(Args));

    if constexpr (sizeof...(Args) > 0)
        push<Args...>(vec);

    return vec;
}
