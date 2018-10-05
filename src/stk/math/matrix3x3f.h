#pragma once

#include <stk/cuda/cuda.h>

#include "types.h"
#include "float3.h"

inline CUDA_HOST_DEVICE Matrix3x3f operator+(const Matrix3x3f& left, const Matrix3x3f& right) {
    return Matrix3x3f {{
        left._rows[0] + right._rows[0],
        left._rows[1] + right._rows[1],
        left._rows[2] + right._rows[2],
    }};
}

inline CUDA_HOST_DEVICE Matrix3x3f operator-(const Matrix3x3f& left, const Matrix3x3f& right) {
    return Matrix3x3f {{
        left._rows[0] - right._rows[0],
        left._rows[1] - right._rows[1],
        left._rows[2] - right._rows[2],
    }};
}

inline CUDA_HOST_DEVICE float3 operator*(const Matrix3x3f& left, const float3& right) {
    return float3({
        dot(left._rows[0], right),
        dot(left._rows[1], right),
        dot(left._rows[2], right)
    });
}

