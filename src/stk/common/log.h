#pragma once

#include <sstream>

//#define STK_LOGGING_DETAILED_PREAMBLE

namespace stk
{
    enum LogLevel
    {
        Debug,
        Info,
        Warning,
        Error, // Typically asserts and such
        Fatal, // Error that kills the application
        Num_LogLevel
    };

    class LogMessage
    {
    public:
        LogMessage(LogLevel level);
        LogMessage(LogLevel level, const char* file, int line);
        ~LogMessage();

        std::ostringstream& stream();

    private:
        void format_preamble(LogLevel level, const char* file = 0, int line = -1);

        LogLevel _level;
        std::ostringstream _s;
    };

    typedef void (LogCallback)(void*, LogLevel, const char*);

    // Initializes logging, should be called at application startup
    void log_init();
    
    // Shuts the logging system down.
    void log_shutdown();

    // Creates a new log file and outputs all messages above the specified level
    void log_add_file(const char* file, LogLevel level);

    // Stops the output to the given file, assuming the file was added with log_add_file
    void log_remove_file(const char* file);

    // Creates a callback for all log messages above the specified level
    void log_add_callback(LogCallback* fn, void* user_data, LogLevel level);

    // Removes the specified callback, as given to log_add_callback
    void log_remove_callback(LogCallback* fn, void* user_data);
}

template<typename T>
stk::LogMessage& operator<<(stk::LogMessage& s, const T& v)
{
    s << v;
    return s;
}

#ifdef STK_LOGGING_DETAILED_PREAMBLE
    #define LOG(level) stk::LogMessage(stk::##level, __FILE__, __LINE__).stream()
#else
    #define LOG(level) stk::LogMessage(stk::##level).stream()
#endif
