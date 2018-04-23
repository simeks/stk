#include "catch.hpp"

#include <stk/common/log.h>

void log_callback(void*, stk::LogLevel, const char* ) {}

TEST_CASE("logging_basic", "[logging]")
{
    stk::log_init();
    
    stk::log_add_file("test_log.txt", stk::Info);
    stk::log_add_callback(log_callback, nullptr, stk::Info);
    
    LOG(Debug) << "A" << "B" << "C";
    LOG(Info) << "A" << "B" << "C";
    LOG(Warning) << "A" << "B" << "C";
    LOG(Error) << "A" << "B" << "C";
    
    stk::log_shutdown();

    REQUIRE(true);
}
