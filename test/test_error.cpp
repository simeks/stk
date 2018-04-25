#include "catch.hpp"

#include <stk/common/error.h>
#include <stk/common/log.h>

namespace
{
    struct LogData
    {
        int last_level;
        std::string last_msg;
    };

    void log_callback(void* user_data, stk::LogLevel level, const char* msg)
    {
        LogData* data = static_cast<LogData*>(user_data);
        data->last_level = level;
        data->last_msg = msg;
    }
}


TEST_CASE("error", "[error] [logging]")
{
    stk::log_init();

    LogData data = { -1, "" };
    stk::log_add_callback(log_callback, &data, stk::Fatal);

#if STK_USE_EXCEPTIONS
    bool catched = false;
    try {
        FATAL() << "this did not work";
    } 
    catch (stk::FatalException& ) {
        catched = true;
    }
    // Assure error was thrown and catched
    REQUIRE(catched);
#else
    REQUIRE(false); // Not implemented
#endif

    // Check log
    REQUIRE(data.last_msg.find("this did not work") != std::string::npos);

    stk::log_remove_callback(log_callback, &data);
    stk::log_shutdown();
}
