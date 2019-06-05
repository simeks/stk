#include "catch.hpp"

#include <stk/common/log.h>

#include <fstream>
#include <string>

using Catch::Matchers::Contains;
using Catch::Matchers::StartsWith;

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

TEST_CASE("logging_basic", "[logging]")
{
    stk::log_init();

    LOG(Info) << "Verbose " << 0;
    LOG(Info) << "Info " << 1;
    LOG(Warning) << "Warning " << 1.0;
    LOG(Error) << "Error " << 'a';
    LOG(Fatal) << "Fatal " << "Fatal " << "Fatal";

    stk::log_shutdown();

    REQUIRE(true);
}

TEST_CASE("logging_callback", "[logging]")
{
    stk::log_init();

    LogData data = { -1, "" };
    stk::log_add_callback(log_callback, &data, stk::Fatal);

    // Should not trigger callback, since Verbose < Fatal
    LOG(Verbose) << "Verbose";
    REQUIRE(data.last_level == -1);
    REQUIRE(data.last_msg == "");

    // Should not trigger callback, since Info < Fatal
    LOG(Info) << "Info";
    REQUIRE(data.last_level == -1);
    REQUIRE(data.last_msg == "");

    // Should not trigger callback, since Warning < Fatal
    LOG(Warning) << "Warning";
    REQUIRE(data.last_level == -1);
    REQUIRE(data.last_msg == "");

    // Should not trigger callback, since Error < Fatal
    LOG(Error) << "Error";
    REQUIRE(data.last_level == -1);
    REQUIRE(data.last_msg == "");

    // Should trigger callback, since Error < Fatal
    LOG(Fatal) << "Fatal";
    REQUIRE(data.last_level == stk::Fatal);
    REQUIRE_THAT(data.last_msg, Contains("Fatal"));

    data = { -1, "" };
    stk::log_remove_callback(log_callback, &data);

    // No logging should trigger the callback since it has been removed
    LOG(Verbose) << "Verbose";
    REQUIRE(data.last_level == -1);
    REQUIRE(data.last_msg == "");

    LOG(Info) << "Info";
    REQUIRE(data.last_level == -1);
    REQUIRE(data.last_msg == "");

    LOG(Warning) << "Warning";
    REQUIRE(data.last_level == -1);
    REQUIRE(data.last_msg == "");

    LOG(Error) << "Error";
    REQUIRE(data.last_level == -1);
    REQUIRE(data.last_msg == "");

    LOG(Fatal) << "Fatal";
    REQUIRE(data.last_level == -1);
    REQUIRE(data.last_msg == "");

    data = { -1, "" };
    stk::log_add_callback(log_callback, &data, stk::Verbose);

    // Should trigger for all messages
    LOG(Verbose) << "Verbose";
    REQUIRE(data.last_level == stk::Verbose);
    REQUIRE_THAT(data.last_msg, Contains("Verbose"));

    LOG(Info) << "Info";
    REQUIRE(data.last_level == stk::Info);
    REQUIRE_THAT(data.last_msg, Contains("Info"));

    LOG(Warning) << "Warning";
    REQUIRE(data.last_level == stk::Warning);
    REQUIRE_THAT(data.last_msg, Contains("Warning"));

    LOG(Error) << "Error";
    REQUIRE(data.last_level == stk::Error);
    REQUIRE_THAT(data.last_msg, Contains("Error"));

    LOG(Fatal) << "Fatal";
    REQUIRE(data.last_level == stk::Fatal);
    REQUIRE_THAT(data.last_msg, Contains("Fatal"));

    stk::log_remove_callback(log_callback, &data);
    stk::log_shutdown();
}


namespace
{
    struct UserType
    {
        const char* str1;
        const char* str2;
    };

    std::ostream& operator<<(std::ostream& out, const UserType& t)
    {
        out << t.str1 << t.str2;
        return out;
    }
}

TEST_CASE("logging_user_type", "[logging]")
{
    stk::log_init();

    LogData data = { -1, "" };
    stk::log_add_callback(log_callback, &data, stk::Info);

    UserType t {"AB", "C"};

    LOG(Info) << t;
    REQUIRE_THAT(data.last_msg, Contains("ABC"));

    stk::log_remove_callback(log_callback, &data);
    stk::log_shutdown();
}

TEST_CASE("logging_file", "[logging]")
{
    stk::log_init();

    // Test 1, only 1 level
    SECTION("fatal level")
    {
        stk::log_add_file("test_logging_file_1.txt", stk::Fatal);

        LOG(Verbose) << "Verbose";
        LOG(Info) << "Info";
        LOG(Warning) << "Warning";
        LOG(Error) << "Error";
        LOG(Fatal) << "Fatal";

        // Remove file to make sure it's flushed and closed
        stk::log_remove_file("test_logging_file_1.txt");

        // Confirm log file
        std::ifstream fs("test_logging_file_1.txt");

        std::string line;
        REQUIRE(std::getline(fs, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("FAT")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Fatal"));
        REQUIRE(!std::getline(fs, line)); // Should only contain one line
    }

    // Test 2, all levels
    SECTION("all levels")
    {
        stk::log_add_file("test_logging_file_2.txt", stk::Verbose);

        LOG(Verbose) << "Verbose";
        LOG(Info) << "Info";
        LOG(Warning) << "Warning";
        LOG(Error) << "Error";
        LOG(Fatal) << "Fatal";

        // Remove file to make sure it's flushed and closed
        stk::log_remove_file("test_logging_file_2.txt");

        // Confirm log file
        std::ifstream fs("test_logging_file_2.txt");

        std::string line;

        REQUIRE(std::getline(fs, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("VER")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Verbose"));

        REQUIRE(std::getline(fs, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("INF")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Info"));

        REQUIRE(std::getline(fs, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("WAR")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Warning"));

        REQUIRE(std::getline(fs, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("ERR")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Error"));

        REQUIRE(std::getline(fs, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("FAT")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Fatal"));

        REQUIRE(!std::getline(fs, line));
    }

    stk::log_shutdown();
}

TEST_CASE("logging_stream", "[logging]")
{
    stk::log_init();

    // Test 1, only 1 level
    SECTION("fatal level")
    {
        std::stringstream ss;
        stk::log_add_stream(&ss, stk::Fatal);

        LOG(Verbose) << "Verbose";
        LOG(Info) << "Info";
        LOG(Warning) << "Warning";
        LOG(Error) << "Error";
        LOG(Fatal) << "Fatal";

        // Test removal
        stk::log_remove_stream(&ss);

        std::string line;
        REQUIRE(std::getline(ss, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("FAT")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Fatal"));
        REQUIRE(!std::getline(ss, line)); // Should only contain one line
    }

    // Test 2, all levels
    SECTION("all levels")
    {
        std::stringstream ss;
        stk::log_add_stream(&ss, stk::Verbose);

        LOG(Verbose) << "Verbose";
        LOG(Info) << "Info";
        LOG(Warning) << "Warning";
        LOG(Error) << "Error";
        LOG(Fatal) << "Fatal";

        // Test removal
        stk::log_remove_stream(&ss);

        std::string line;

        REQUIRE(std::getline(ss, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("VER")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Verbose"));

        REQUIRE(std::getline(ss, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("INF")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Info"));

        REQUIRE(std::getline(ss, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("WAR")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Warning"));

        REQUIRE(std::getline(ss, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("ERR")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Error"));

        REQUIRE(std::getline(ss, line));
        #ifdef STK_LOGGING_PREFIX_LEVEL
            REQUIRE_THAT(line, StartsWith("FAT")); // From prefix
        #endif
        #ifdef STK_LOGGING_PREFIX_FILE
            REQUIRE_THAT(line, Contains("test_logging.cpp")); // From prefix
        #endif
        REQUIRE_THAT(line, Contains("Fatal"));

        REQUIRE(!std::getline(ss, line));
    }

    stk::log_shutdown();
}

TEST_CASE("logging_file_and_callback", "[logging]")
{
    // Check if both file and callback output works simultaneously

    stk::log_init();

    LogData data = { -1, "" };
    stk::log_add_callback(log_callback, &data, stk::Verbose);
    stk::log_add_file("test_logging_file_3.txt", stk::Verbose);

    LOG(Verbose) << "Verbose";
    REQUIRE(data.last_level == stk::Verbose);
    REQUIRE_THAT(data.last_msg, Contains("Verbose"));

    LOG(Info) << "Info";
    REQUIRE(data.last_level == stk::Info);
    REQUIRE_THAT(data.last_msg, Contains("Info"));

    LOG(Warning) << "Warning";
    REQUIRE(data.last_level == stk::Warning);
    REQUIRE_THAT(data.last_msg, Contains("Warning"));

    LOG(Error) << "Error";
    REQUIRE(data.last_level == stk::Error);
    REQUIRE_THAT(data.last_msg, Contains("Error"));

    LOG(Fatal) << "Fatal";
    REQUIRE(data.last_level == stk::Fatal);
    REQUIRE_THAT(data.last_msg, Contains("Fatal"));

    std::ifstream fs("test_logging_file_3.txt");

    std::string line;
    REQUIRE(std::getline(fs, line));
    REQUIRE_THAT(line, Contains("Verbose"));
    REQUIRE(std::getline(fs, line));
    REQUIRE_THAT(line, Contains("Info"));
    REQUIRE(std::getline(fs, line));
    REQUIRE_THAT(line, Contains("Warning"));
    REQUIRE(std::getline(fs, line));
    REQUIRE_THAT(line, Contains("Error"));
    REQUIRE(std::getline(fs, line));
    REQUIRE_THAT(line, Contains("Fatal"));

    stk::log_shutdown();
}

TEST_CASE("logging_debug", "[logging]")
{
    DLOG(Info) << "A" << "B" << "C";
    REQUIRE(true);
}
