#pragma once

#if defined(_WIN32) && defined(_MSC_VER)
    #define STK_ALIGN(n) __declspec(align(n))
#elif defined(__GNUC__)
    #define STK_ALIGN(n) __attribute__((aligned(n)))
#endif
