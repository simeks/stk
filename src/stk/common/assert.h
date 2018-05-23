#pragma once

#include "error.h"

// Assertions are used as you would expect.
// ASSERT(expr) triggers whenever expr is false. A triggered assertion will log
//  the about failed assertion and abort the application.
// ASSERT will be included in all builds while DASSERT will only be included in
//  debug builds.

#define ASSERT(expr) FATAL_IF(!(expr)) << "Assertion failed: " # expr

#ifndef NDEBUG
    #define DASSERT(expr) FATAL_IF(!(expr)) << "Assertion failed: " # expr
#else
    #define DASSERT(expr)
#endif
