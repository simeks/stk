#include "log.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
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
        enum Type { Type_FileSink, Type_CallbackSink };

        Sink(stk::LogLevel level) : _level(level) {}

        virtual void write(stk::LogLevel level, const char* msg) = 0;
        virtual Type type() const = 0;

    protected:
        stk::LogLevel _level;
    };

    class FileSink : public Sink
    {
    public:
        FileSink(stk::LogLevel level) : Sink(level)
        {
        }
        ~FileSink()
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
        void write(stk::LogLevel, const char* msg)
        {
            std::lock_guard<std::mutex> guard(_lock);

            if (_fs.is_open()) {
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

    struct LoggerData
    {
        std::vector<Sink*> sinks;
    };

    LoggerData* _logger_data = nullptr;

    void log_message(stk::LogLevel level, const std::string& msg)
    {
        if (!_logger_data) {
            return;
        }

        for (auto& s : _logger_data->sinks) {
            s->write(level, msg.c_str());
        }
        
        if (level >= stk::Error) {
            std::cerr << msg;
        }
        else {
            std::cout << msg;
        }

        if (level == stk::Fatal) {
            abort();
        }
    }
}

namespace stk
{
    LogMessage::LogMessage(LogLevel level) :
        _level(level)
    {
        format_preamble(level);
    }
    LogMessage::LogMessage(LogLevel level, const char* file, int line) :
        _level(level)
    {
        format_preamble(level, file, line);
    }
    LogMessage::~LogMessage()
    {
        _s << std::endl;
        // Flush
        log_message(_level, _s.str());
    }
    std::ostringstream& LogMessage::stream()
    {
        return _s;
    }
    void LogMessage::format_preamble(LogLevel level, const char* file, int line)
    {
        auto now = system_clock::now();
        auto in_time = system_clock::to_time_t(now);
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        const char* level_to_str[Num_LogLevel] = {
            "D",
            "I",
            "W",
            "E",
            "F", 
        };

        _s << "[" << level_to_str[level] << " " << std::put_time(std::localtime(&in_time), "%m-%d %X") << "." << ms.count();

        if (file && line >= 0) {
            _s << " " << file << ":" << line << "] ";
        }
        else {
            _s << "] ";
        }
    }


    void log_init()
    {
        _logger_data = new LoggerData();
    }
    void log_shutdown()
    {
        for (auto& s : _logger_data->sinks) {
            delete s;
        }
        
        delete _logger_data;
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
            if ((*it)->type() == Sink::Type_FileSink &&
                static_cast<CallbackSink*>(*it)->fn() == fn &&
                static_cast<CallbackSink*>(*it)->user_data() == user_data)
            {
                _logger_data->sinks.erase(it);
                return;
            }
        }
    }
}