#include "error.h"
#include "log.h"


namespace stk
{

#if STK_USE_EXCEPTIONS
    FatalException::FatalException(const char* message) : _message(message) {}
    FatalException::~FatalException() throw() {}

    const char* FatalException::what() const throw()
    {
        return _message.c_str();
    }
#endif // STK_USE_EXCEPTIONS

    FatalError::FatalError(const char* file, int line) :
        _file(file), 
        _line(line)
    {
        _s << "Fatal error: ";
    }
    FatalError::~FatalError() noexcept(false)
    {
        const char* file = nullptr;
        for (const char* ptr = _file; *ptr; ++ptr) {
            if (*ptr == '/' || *ptr == '\\') {
                file = ptr + 1;
            }
        }

        _s << " (" << file << ":" << _line << ")";

        // We do not use the macro as we want to make sure we have the
        //  file name and line number of the call to FATAL().
        LogMessage(Fatal, _file, _line).stream() << _s.str();

    #ifdef STK_USE_EXCEPTIONS
        // Should probably be careful about throwing exceptions from a destructor,
        //  however, this class will not be used in any containers and if we happen
        //  to already have an active exception, it is probably triggered by another
        //  FATAL()-invocation.
        throw FatalException(_s.str().c_str());
    #else 
        abort();
    #endif
    }
    std::ostringstream& FatalError::stream()
    {
        return _s;
    }
}
