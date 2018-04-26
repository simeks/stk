#include "catch.hpp"

#include <stk/common/assert.h>
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

using Catch::Matchers::Contains;

TEST_CASE("error", "[error] [logging]")
{
    stk::log_init();

    LogData data = { -1, "" };
    stk::log_add_callback(log_callback, &data, stk::Fatal);

#if STK_USE_EXCEPTIONS
    REQUIRE_THROWS_WITH(FATAL() << "this did not work", Contains("this did not work"));
#else
    REQUIRE(false); // Not implemented
#endif

    // Check log
    REQUIRE(data.last_level == stk::Fatal);
    REQUIRE_THAT(data.last_msg, Contains("this did not work"));

    stk::log_remove_callback(log_callback, &data);
    stk::log_shutdown();
}

TEST_CASE("error_if", "[error] [logging]")
{
    stk::log_init();

#if STK_USE_EXCEPTIONS
    SECTION("true")
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        REQUIRE_THROWS_WITH(FATAL_IF(true) << "this did not work", Contains("this did not work"));

        // Check log
        REQUIRE(data.last_level == stk::Fatal);
        REQUIRE_THAT(data.last_msg, Contains("this did not work"));

        stk::log_remove_callback(log_callback, &data);
    }
    SECTION("false")
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        REQUIRE_NOTHROW(FATAL_IF(false) << "this did not work");

        REQUIRE(data.last_level == -1);
        REQUIRE(data.last_msg == "");

        stk::log_remove_callback(log_callback, &data);
    }
#else
    REQUIRE(false); // Not implemented
#endif

    stk::log_shutdown();
}

TEST_CASE("error_assert", "[error] [logging]")
{
    stk::log_init();

#if STK_USE_EXCEPTIONS
    SECTION("false") // Should trigger
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        // Ensure error was thrown and catched
        REQUIRE_THROWS_WITH(ASSERT(false), Contains("Assertion failed: false"));

        // Check log
        REQUIRE(data.last_level == stk::Fatal);
        REQUIRE_THAT(data.last_msg, Contains("Assertion failed: false"));

        stk::log_remove_callback(log_callback, &data);
    }
    SECTION("true") // Should not trigger
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        // Error should not have been triggered
        REQUIRE_NOTHROW(ASSERT(true));

        REQUIRE(data.last_level == -1);
        REQUIRE(data.last_msg == "");

        stk::log_remove_callback(log_callback, &data);
    }
#else
    REQUIRE(false); // Not implemented
#endif

    stk::log_shutdown();
}

#ifndef NDEBUG
TEST_CASE("error_dassert", "[error] [logging]")
{
    stk::log_init();

#if STK_USE_EXCEPTIONS
    SECTION("false") // Should trigger
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        // Assure error was thrown and catched
        REQUIRE_THROWS_WITH(DASSERT(false), Contains("Assertion failed: false"));

        // Check log
        REQUIRE(data.last_level == stk::Fatal);
        REQUIRE_THAT(data.last_msg, Contains("Assertion failed: false"));

        stk::log_remove_callback(log_callback, &data);
    }
    SECTION("true") // Should not trigger
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        // Error should not have been triggered
        REQUIRE_NOTHROW(DASSERT(true));
        
        // Check log
        REQUIRE(data.last_level == -1);
        REQUIRE(data.last_msg == "");

        stk::log_remove_callback(log_callback, &data);
    }
#else
    REQUIRE(false); // Not implemented
#endif

    stk::log_shutdown();
}
#endif
