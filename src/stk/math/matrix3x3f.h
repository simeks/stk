#pragma once

#include <stk/cuda/cuda.h>

#include "types.h"
#include "float3.h"

inline CUDA_HOST_DEVICE float3 operator*(const Matrix3x3f& left, const float3& right) {
    return float3({
        dot(left._rows[0], right),
        dot(left._rows[1], right),
        dot(left._rows[2], right)
    });
}

