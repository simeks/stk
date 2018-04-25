#include "catch.hpp"

#include <stk/common/log.h>

#include <fstream>
#include <string>

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
    REQUIRE(data.last_msg.find("Fatal") != std::string::npos);

    data = { -1, "" };
    stk::log_remove_callback(log_callback, &data);

    // No logging should trigger the callback since it has been removed
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
    stk::log_add_callback(log_callback, &data, stk::Info);

    // Should trigger for all messages
    LOG(Info) << "Info";
    REQUIRE(data.last_level == stk::Info);
    REQUIRE(data.last_msg.find("Info") != std::string::npos);

    LOG(Warning) << "Warning";
    REQUIRE(data.last_level == stk::Warning);
    REQUIRE(data.last_msg.find("Warning") != std::string::npos);
    
    LOG(Error) << "Error";
    REQUIRE(data.last_level == stk::Error);
    REQUIRE(data.last_msg.find("Error") != std::string::npos);
    
    LOG(Fatal) << "Fatal";
    REQUIRE(data.last_level == stk::Fatal);
    REQUIRE(data.last_msg.find("Fatal") != std::string::npos);

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
    REQUIRE(data.last_msg.find("ABC") != std::string::npos);

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
        #ifdef STK_LOGGING_PREAMBLE_PRINT_LEVEL
            REQUIRE(line.find("FAT") == 0); // From preamble
        #endif
        #ifdef STK_LOGGING_PREAMBLE_PRINT_FILE
            REQUIRE(line.find("test_logging.cpp") != std::string::npos); // From preamble
        #endif
        REQUIRE(line.find("Fatal") != std::string::npos);
        REQUIRE(!std::getline(fs, line)); // Should only contain one line
    }
    
    // Test 2, all levels
    SECTION("all levels")
    {
        stk::log_add_file("test_logging_file_2.txt", stk::Info);

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
        #ifdef STK_LOGGING_PREAMBLE_PRINT_LEVEL
            REQUIRE(line.find("INF") == 0); // From preamble
        #endif
        #ifdef STK_LOGGING_PREAMBLE_PRINT_FILE
            REQUIRE(line.find("test_logging.cpp") != std::string::npos); // From preamble
        #endif
        REQUIRE(line.find("Info") != std::string::npos);
        
        REQUIRE(std::getline(fs, line));
        #ifdef STK_LOGGING_PREAMBLE_PRINT_LEVEL
            REQUIRE(line.find("WAR") == 0); // From preamble
        #endif
        #ifdef STK_LOGGING_PREAMBLE_PRINT_FILE
            REQUIRE(line.find("test_logging.cpp") != std::string::npos); // From preamble
        #endif
        REQUIRE(line.find("Warning") != std::string::npos);
        
        REQUIRE(std::getline(fs, line));
        #ifdef STK_LOGGING_PREAMBLE_PRINT_LEVEL
            REQUIRE(line.find("ERR") == 0); // From preamble
        #endif
        #ifdef STK_LOGGING_PREAMBLE_PRINT_FILE
            REQUIRE(line.find("test_logging.cpp") != std::string::npos); // From preamble
        #endif
        REQUIRE(line.find("Error") != std::string::npos);
        
        REQUIRE(std::getline(fs, line));
        #ifdef STK_LOGGING_PREAMBLE_PRINT_LEVEL
            REQUIRE(line.find("FAT") == 0); // From preamble
        #endif
        #ifdef STK_LOGGING_PREAMBLE_PRINT_FILE
            REQUIRE(line.find("test_logging.cpp") != std::string::npos); // From preamble
        #endif
        REQUIRE(line.find("Fatal") != std::string::npos);
        
        REQUIRE(!std::getline(fs, line));
    }

    stk::log_shutdown();
}

TEST_CASE("logging_debug", "[logging]")
{
    DLOG(Info) << "A" << "B" << "C";
    REQUIRE(true);
}
