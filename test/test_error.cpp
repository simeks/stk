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

TEST_CASE("error_if", "[error] [logging]")
{
    stk::log_init();

#if STK_USE_EXCEPTIONS
    SECTION("true")
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        bool thrown = false;
        try {
            FATAL_IF(true) << "this did not work";
        }
        catch (stk::FatalException& ) {
            thrown = true;
        }
        // Assure error was thrown and catched
        REQUIRE(thrown);

        // Check log
        REQUIRE(data.last_msg.find("this did not work") != std::string::npos);

        stk::log_remove_callback(log_callback, &data);
    }
    SECTION("false")
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        bool thrown = false;
        try {
            FATAL_IF(false) << "this did not work";
        }
        catch (stk::FatalException& ) {
            thrown = true;
        }
        // Error should not have been triggered
        REQUIRE(!thrown);

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

        bool thrown = false;
        try {
            ASSERT(false);
        }
        catch (stk::FatalException& ) {
            thrown = true;
        }
        // Assure error was thrown and catched
        REQUIRE(thrown);

        // Check log
        REQUIRE(data.last_msg.find("Assertion failed: false") != std::string::npos);

        stk::log_remove_callback(log_callback, &data);
    }
    SECTION("true") // Should not trigger
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        bool thrown = false;
        try {
            ASSERT(true);
        }
        catch (stk::FatalException& ) {
            thrown = true;
        }
        // Error should not have been triggered
        REQUIRE(!thrown);

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

        bool thrown = false;
        try {
            DASSERT(false);
        }
        catch (stk::FatalException& ) {
            thrown = true;
        }
        // Assure error was thrown and catched
        REQUIRE(thrown);

        // Check log
        REQUIRE(data.last_msg.find("Assertion failed: false") != std::string::npos);

        stk::log_remove_callback(log_callback, &data);
    }
    SECTION("true") // Should not trigger
    {
        LogData data = { -1, "" };
        stk::log_add_callback(log_callback, &data, stk::Fatal);

        bool thrown = false;
        try {
            DASSERT(true);
        }
        catch (stk::FatalException& ) {
            thrown = true;
        }
        // Error should not have been triggered
        REQUIRE(!thrown);
        
        // Check log
        REQUIRE(data.last_msg == "");

        stk::log_remove_callback(log_callback, &data);
    }
#else
    REQUIRE(false); // Not implemented
#endif

    stk::log_shutdown();
}
#endif
