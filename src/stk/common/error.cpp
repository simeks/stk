#include "error.h"
#include "log.h"

namespace
{
    void fail()
    {
        exit(1);
    }

    void (*_fail_fn)();
}

void stk::error(const char* msg, const char* file, int line)
{
    LOG(Fatal) << "Error: " << msg << " (" << file << ":" << line << ")";
    exit(1);
}