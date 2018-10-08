#pragma once

#include <stk/image/volume.h>

namespace stk
{
    // Normalizes a scalar volume to the specified range
    // An optional output volume parameter ('out') can be passed to the function.
    //  If provided, the results will be written to 'out'. This assumes that 'out'
    //  has the same dimensions as 'src'. If not, a new volume is allocated.
    template<typename T>
    VolumeHelper<T> normalize(
        const VolumeHelper<T>& src,
        T min,
        T max,
        VolumeHelper<T>* out=nullptr
    );
}

#include "normalize.inl"
