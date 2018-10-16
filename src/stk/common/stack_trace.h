#pragma once

#include <sstream>

#if defined(__GNUC__) && defined(__linux__)

// Relying on GNU extensions
#include <execinfo.h>
#include <unistd.h>
#include <stdlib.h>

void stack_trace(std::ostringstream& s)
{
    #define BT_BUF_SIZE 1024
    void *buffer[BT_BUF_SIZE];
    int bt_len = backtrace(buffer, BT_BUF_SIZE);
    char **bt = backtrace_symbols(buffer, bt_len);
    s << std::endl;
    for (int i = 0; i < bt_len; ++i) {
        s << std::string (bt[i]) << std::endl;
    }
    free(bt);
    #undef BT_BUF_SIZE
}

#else // defined(__GNUC__) && defined(__linux__)

// FIXME to implement something equivalent for MSVC
void stack_trace(std::ostringsream& /* s */)
{
}

#endif // defined(__GNUC__) && defined(__linux__)

