#include "stack_trace.h"

#if defined(__linux__) && defined(STK_USE_BACKWARD)

#include <backward.hpp>

void get_stack_trace(std::ostream& s, const int skip)
{
    backward::StackTrace st;
    st.load_here(128);
    st.skip_n_firsts(1 + skip); // skip current frame
    backward::Printer p;
    p.object = p.address = true;
    p.color_mode = backward::ColorMode::always;
    p.print(st, s);
}

#elif  defined(__GNUC__) && defined(__linux__)

// Linux-only, relying on GNU extensions
#include <execinfo.h>
#include <unistd.h>
#include <stdlib.h>

void get_stack_trace(std::ostream& s, const int skip)
{
    #define BT_BUF_SIZE 1024
    void *buffer[BT_BUF_SIZE];
    int bt_len = backtrace(buffer, BT_BUF_SIZE);
    char **bt = backtrace_symbols(buffer, bt_len);
    for (int i = 1 + skip; i < bt_len; ++i) { // skip current frame
        s << "# " << i - skip << " " << std::string (bt[i]) << std::endl;
    }
    free(bt);
    #undef BT_BUF_SIZE
}

#elif defined(_WIN32)

#include <process.h>
#include <iostream>
#include <sstream>
#include <Windows.h>
#include "dbghelp.h"

#pragma comment ( lib, "dbghelp.lib" )

int get_stack_trace(std::ostream& s, const int skip)
{
    #define TRACE_MAX_STACK_FRAMES 256
    #define TRACE_MAX_FUNCTION_NAME_LENGTH 256

    HANDLE process = GetCurrentProcess();
    SymInitialize(process, NULL, TRUE);

    // Always skip the first frame (this function)
    void* stack[TRACE_MAX_STACK_FRAMES];
    WORD stack_len = CaptureStackBackTrace(1 + skip, TRACE_MAX_STACK_FRAMES, stack, NULL);

    char buffer[sizeof(SYMBOL_INFO) + (TRACE_MAX_FUNCTION_NAME_LENGTH - 1) * sizeof(TCHAR)];
    SYMBOL_INFO * const symbol = (SYMBOL_INFO*) &buffer;
    IMAGEHLP_LINE64 line = { sizeof(IMAGEHLP_LINE64), 0, 0, 0 };

    symbol->MaxNameLen = TRACE_MAX_FUNCTION_NAME_LENGTH;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    for (int i = 0; i < stack_len; i++) {
        DWORD displacement;
        DWORD64 address = (DWORD64) stack[i];
        SymFromAddr(process, address, NULL, symbol);
        BOOL found = SymGetLineFromAddr64(process, address, &displacement, &line);
        s << "# " << i + 1
                  << "\tSource \"" << (found ? line.FileName : "??")
                  << "\", line " << (found ? std::to_string(line.LineNumber) : "??")
                  << ":" << (found ? std::to_string(displacement) : "??")
                  << ", in " << symbol->Name
                  << " [0x" <<  std::hex << symbol->Address << std::dec << "]" << std::endl;
    }

    SymCleanup(process);

    #undef TRACE_MAX_STACK_FRAMES
    #undef TRACE_MAX_FUNCTION_NAME_LENGTH

    return 0;
}

# else // defined(__GNUC__) && defined(__linux__)

int get_stack_trace(std::ostream& s, const int skip)
{
    (void) s;
    (void) skip;
}

#endif // defined(__GNUC__) && defined(__linux__)

