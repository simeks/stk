#pragma once

#include <sstream>

#include "log.h"

#define STK_USE_EXCEPTIONS 1

#if STK_USE_EXCEPTIONS
    #include <exception>
    #include <string>
#endif

// TODO:
// There seem to be something weird going on with 'noreturn' on gcc 4 or 5. Works fine on
//  gcc-9.
// Example:
// try {
//     FATAL_IF(true) << "asd";
// } catch (...) {
// }
// results in : "terminate called after throwing an instance of 'stk::FatalException'".
// I.e. the application seems to terminate before passing the exception further.
#if defined __GNUC__ && __GNUC__ <= 5
    #pragma GCC diagnostic ignored "-Wreturn-type"
    #define STK_NORETURN
#else
    #define STK_NORETURN [[noreturn]]
#endif

namespace stk
{
#if STK_USE_EXCEPTIONS
    class FatalException : public std::exception
    {
    public:
        FatalException(const char* message);
        virtual ~FatalException() throw();

        virtual const char* what() const throw();

    private:
        std::string _message;
    };
#endif

    // Helper class for the FATAL()-macro
    class FatalError
    {
    public:
        FatalError(const char* file, int line);
        STK_NORETURN ~FatalError() noexcept(false);

        std::ostringstream& stream();
    private:
        std::ostringstream _s;

        const char* _file;
        int _line;
    };


    // Logs the error message and either exits the application or throws an 
    //  exception depending on whether STK_USE_EXCEPTIONS is set or not.
    // Typically called through any of the error macros.
    void error(const char* msg, const char* file, int line);
}

// Usage: FATAL() << "Error message";
#define FATAL() stk::FatalError(__FILE__, __LINE__).stream()

// Usage: FATAL_IF(failed==true) << "Error message";
#define FATAL_IF(expr) !(expr) ? (void)0 : stk::LogFinisher() & FATAL()

// Just a macro to identify "TODOs" in code
#define NOT_IMPLEMENTED() FATAL()
