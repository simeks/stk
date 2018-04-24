#pragma once

namespace stk
{
    // Logs the error message and exits the application.
    // Typically called through any of the error macros.
    void error(const char* msg, const char* file, int line);
}

