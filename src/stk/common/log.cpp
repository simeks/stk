#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

namespace
{
    class Sink
    {
    public:
        enum Type { Type_FileSink, Type_CallbackSink, Type_StreamSink };

        Sink(stk::LogLevel level) : _level(level) {}
        virtual ~Sink() {};

        virtual void write(stk::LogLevel level, const char* msg) = 0;
        virtual void flush() {}

        virtual Type type() const = 0;

        stk::LogLevel level() { return _level; }

    protected:
        stk::LogLevel _level;
    };

    class FileSink : public Sink
    {
    public:
        FileSink(stk::LogLevel level) : Sink(level)
        {
        }
        virtual ~FileSink()
        {
            flush();
            _fs.close();
        }
        void open(const char* file)
        {
            if(_fs.is_open()) {
                _fs.close();
            }

            _fs.open(file, std::ios::out);

            // Using our own assert/logging functionality here will cause a deadlock
            if(!_fs.is_open()) {
                std::cerr << "Failed to open log file: '" << file << "'" << std::endl;
                return;
            }
            _file = file;
        }
        void write(stk::LogLevel level, const char* msg)
        {
            std::lock_guard<std::mutex> guard(_lock);

            if(_fs.is_open() && level >= _level) {
                _fs.write(msg, strlen(msg));
            }
        }
        void flush()
        {
            std::lock_guard<std::mutex> guard(_lock);
            _fs.flush();
        }

        const std::string& file() const
        {
            return _file;
        }

        Type type() const
        {
            return Type_FileSink;
        }

    private:
        std::mutex _lock;
        std::ofstream _fs;
        std::string _file;
    };

    class CallbackSink : public Sink
    {
    public:
        CallbackSink(stk::LogCallback* fn, void* user_data, stk::LogLevel level) :
            Sink(level),
            _fn(fn),
            _user_data(user_data)
        {
        }

        void write(stk::LogLevel level, const char* msg)
        {
            if(_fn && level >= _level) {
                _fn(_user_data, level, msg);
            }
        }

        stk::LogCallback* fn() const
        {
            return _fn;
        }
        void* user_data() const
        {
            return _user_data;
        }

        Type type() const
        {
            return Type_CallbackSink;
        }

    private:
        stk::LogCallback* _fn;
        void* _user_data;
    };

    class StreamSink : public Sink
    {
    public:
        StreamSink(std::ostream *os, stk::LogLevel level)
            : Sink(level)
            , _os(os)
        {
        }
        virtual ~StreamSink()
        {
            flush();
        }
        void write(stk::LogLevel level, const char* msg)
        {
            std::lock_guard<std::mutex> guard(_lock);

            if(level >= _level) {
                _os->write(msg, strlen(msg));
            }
        }
        void flush()
        {
            std::lock_guard<std::mutex> guard(_lock);
            _os->flush();
        }

        std::ostream* stream(void)
        {
            return _os;
        }

        Type type() const
        {
            return Type_StreamSink;
        }

    private:
        std::mutex _lock;
        std::ostream *_os;
        std::string _file;
    };

    struct LoggerData
    {
        std::vector<Sink*> sinks;
    };

    LoggerData* _logger_data = nullptr;

    void log_write(stk::LogLevel level, const char* msg)
    {
        if (_logger_data) {
            for (auto& s : _logger_data->sinks) {
                s->write(level, msg);
            }
            if (level == stk::Fatal) {
                // Flush all sinks
                for (auto& s : _logger_data->sinks) {
                    s->flush();
                }
            }
        }
    }
} // namespace

namespace stk
{
    LogMessage::LogMessage(LogLevel level) :
        _level(level),
        _flushed(false)
    {
        format_prefix(level);
    }
    LogMessage::LogMessage(LogLevel level, const char* file, int line) :
        _level(level),
        _flushed(false)
    {
        format_prefix(level, file, line);
    }
    LogMessage::~LogMessage()
    {
        flush();
    }
    std::ostringstream& LogMessage::stream()
    {
        return _s;
    }
    void LogMessage::flush()
    {
        // We only flush messages once
        if (_flushed)
            return;

        _s << std::endl;
        // Flush
        log_write(_level, _s.str().c_str());

        _flushed = true;
    }
    void LogMessage::format_prefix(LogLevel level, const char* file, int line)
    {
    #ifdef STK_LOGGING_PREFIX_LEVEL
        const char* level_to_str[Num_LogLevel] = {
            "VER",
            "INF",
            "WAR",
            "ERR",
            "FAT",
        };
        _s << level_to_str[level] << " ";
    #endif

    #ifdef STK_LOGGING_PREFIX_TIME
        auto now = system_clock::now();
        auto in_time = system_clock::to_time_t(now);
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%m-%d %X", std::localtime(&in_time));

        _s << timestamp << "." << std::setw(3) << std::setfill('0') << ms.count() << " ";
    #endif

    #ifdef STK_LOGGING_PREFIX_FILE
        if (file && line >= 0) {
            // Only include filename
            for (const char* ptr = file; *ptr; ++ptr) {
                if (*ptr == '/' || *ptr == '\\') {
                    file = ptr + 1;
                }
            }
            _s << "[" << file << ":" << std::setw(4) << line << "] ";
        }
    #else
        (void) file; (void) line;
    #endif

        _s << "| ";
    }

    void log_init()
    {
        _logger_data = new LoggerData();
    }
    void log_shutdown()
    {
        if (!_logger_data) {
            return;
        }

        for (auto& s : _logger_data->sinks) {
            delete s;
        }

        delete _logger_data;
        _logger_data = nullptr;
    }
    void log_add_file(const char* file, LogLevel level)
    {
        if (!_logger_data) {
            return; // TODO: Assert!
        }

        FileSink* sink = new FileSink(level);
        sink->open(file);

        _logger_data->sinks.push_back(sink);
    }
    void log_remove_file(const char* file)
    {
        if (!_logger_data) {
            return; // TODO: Assert!
        }

        auto it = _logger_data->sinks.begin();
        for (; it != _logger_data->sinks.end(); ++it)
        {
            if ((*it)->type() == Sink::Type_FileSink &&
                static_cast<FileSink*>(*it)->file() == file)
            {
                delete (*it);
                _logger_data->sinks.erase(it);
                return;
            }
        }
    }
    void log_add_callback(LogCallback* fn, void* user_data, LogLevel level)
    {
        if (!_logger_data) {
            return; // TODO: Assert!
        }

        _logger_data->sinks.push_back(new CallbackSink(fn, user_data, level));
    }
    void log_remove_callback(LogCallback* fn, void* user_data)
    {
        if (!_logger_data) {
            return; // TODO: Assert!
        }

        auto it = _logger_data->sinks.begin();
        for (; it != _logger_data->sinks.end(); ++it)
        {
            if ((*it)->type() == Sink::Type_CallbackSink &&
                static_cast<CallbackSink*>(*it)->fn() == fn &&
                static_cast<CallbackSink*>(*it)->user_data() == user_data)
            {
                delete (*it);
                _logger_data->sinks.erase(it);
                return;
            }
        }
    }
    void log_add_stream(std::ostream * const os, LogLevel level)
    {
        if (!_logger_data) {
            return; // TODO: Assert!
        }

        StreamSink* sink = new StreamSink(os, level);

        _logger_data->sinks.push_back(sink);
    }
    void log_remove_stream(std::ostream * const os)
    {
        if (!_logger_data) {
            return; // TODO: Assert!
        }

        _logger_data->sinks.erase(
                std::remove_if(
                    _logger_data->sinks.begin(),
                    _logger_data->sinks.end(),
                    [&](Sink* s) { return s->type() == Sink::Type_StreamSink &&
                                          static_cast<StreamSink*>(s)->stream() == os; }
                    )
                );
    }
    LogLevel log_level()
    {
        // Default value when no sink is registered
        if (!_logger_data) {
            return LogLevel::Num_LogLevel;
        }
        // Minimum value among the registered sinks
        return std::accumulate(
                _logger_data->sinks.begin(),
                _logger_data->sinks.end(),
                LogLevel::Num_LogLevel,
                [](LogLevel a, Sink *b) { return std::min(a, b->level()); }
                );
    }
    LogLevel log_level_from_str(const std::string& s)
    {
        if (s == "Verbose") {
            return LogLevel::Verbose;
        }
        else if (s == "Info") {
            return LogLevel::Info;
        }
        else if (s == "Warning") {
            return LogLevel::Warning;
        }
        else if (s == "Error") {
            return LogLevel::Error;
        }
        else if (s == "Fatal") {
            return LogLevel::Fatal;
        }
        throw std::runtime_error(("Unrecognised log level '" + s + "'").c_str());
    }
    LogLevel log_level_from_str(const char * const s)
    {
        if (!s) {
            throw std::runtime_error("Invalid log level");
        }
        return log_level_from_str(std::string(s));
    }
}
