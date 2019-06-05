#pragma once

#include <sstream>

#define STK_LOGGING_PREFIX_LEVEL
#define STK_LOGGING_PREFIX_TIME


// Logging utilities
//
// Usage:
//  Initialize the logging using log_init. This step is optional if you only want
//  output to stderr. If you want to use log files or callbacks however, the initialization
//  is required.
//
//  After the initialization you can create log files using log_add_file. This will create
//  or overwrite a file at the specified path and all output above the specified level will
//  be written to said file. E.g. log_add_file('log.txt', stk::Info) will output _all_ log
//  messages to log.txt. To only output error messages, change stk::Info to stk::Error.
//
//  In addition to file outputs you can also add C-style callback functions using
//  log_add_callback. Whenever a log message is received the message (including prefix)
//  will be passed to the callback.
//
//  Opened log files can be closed using log_remove_file and callbacks can be removed with
//  log_remove_callback.
//
//  The LOG() macro is used for logging messages. You specify the level (or severity) of
//  the message within the macro, e.g. LOG(Info), or LOG(Error). A C++-style stream object
//  is returned, allowing logging in the style of LOG(Info) << "My Message" << my_variable;
//
//  For debug message only wanted in debug builds it is advised to use the DLOG() macro.
//  These macros will not be compiled in release builds.
//
//  log_shutdown() cleans up data allocated by the system and closes all open log files.
//
// There are 5 severity levels:
//  * Verbose
//  * Info
//  * Warning
//  * Error
//  * Fatal
//
//  The goal of this subsystem is only to do logging, None of the levels, including Fatal,
//  affects the runtime in any way. Therefore the use of the Fatal level should be limited
//  to use within the error handling system (see error.h) to avoid confusion.


namespace stk
{
    enum LogLevel
    {
        Verbose,
        Info,
        Warning,
        Error,
        Fatal,
        Num_LogLevel
    };

    class LogMessage
    {
    public:
        LogMessage(LogLevel level);
        LogMessage(LogLevel level, const char* file, int line);
        ~LogMessage();

        std::ostringstream& stream();

        void flush();

    private:
        void format_prefix(LogLevel level, const char* file = 0, int line = -1);

        LogLevel _level;
        std::ostringstream _s;

        bool _flushed;
    };

    // Class to nullify incoming log messages, used for debug messages in release build
    class NullStream
    {
    public:
        template<typename T>
        NullStream& operator<<(const T&) { return *this; }
    };

    // Used by the LOG macros to make sure they have a void return.
    // Avoids certain warnings and errors
    struct LogFinisher
    {
        LogFinisher() {}
        void operator&(std::ostream&) {}
    };

    typedef void (LogCallback)(void*, LogLevel, const char*);

    // Initializes logging, should be called at application startup
    void log_init();

    // Shuts the logging system down.
    void log_shutdown();

    // Get the minimum level among the registered sinks
    LogLevel log_level();

    // Creates a new log file and outputs all messages above the specified level
    void log_add_file(const char* file, LogLevel level);

    // Stops the output to the given file, assuming the file was added with log_add_file
    void log_remove_file(const char* file);

    // Creates a callback for all log messages above the specified level
    void log_add_callback(LogCallback* fn, void* user_data, LogLevel level);

    // Removes the specified callback, as given to log_add_callback
    void log_remove_callback(LogCallback* fn, void* user_data);

    // Outputs all messages above the specified level to the given stream
    void log_add_stream(std::ostream * const os, LogLevel level);

    // Stops the output to the given stream, assuming the stream was added with log_add_stream
    void log_remove_stream(std::ostream * const os);

    // Convert a string to a log level
    // If the input is unrecognised, the default value LogLevel::Info is returned
    LogLevel log_level_from_str(const std::string& s);
    LogLevel log_level_from_str(const char * const s);
}

template<typename T>
stk::LogMessage& operator<<(stk::LogMessage& s, const T& v)
{
    s << v;
    return s;
}

#if STK_LOGGING_PREFIX_FILE
    #define _STK_LOG(level) stk::LogFinisher() & stk::LogMessage(stk::level, __FILE__, __LINE__).stream()
#else
    #define _STK_LOG(level) stk::LogFinisher() & stk::LogMessage(stk::level).stream()
#endif

#define LOG(level) (stk::level < stk::log_level()) ? (void)0 : _STK_LOG(level)

#ifdef NDEBUG
    #define DLOG(level) stk::NullStream()
#else
    #define DLOG(level) LOG(level)
#endif

#define LOG_IF(level, expr) !(expr) ? (void)0 : LOG(level)

