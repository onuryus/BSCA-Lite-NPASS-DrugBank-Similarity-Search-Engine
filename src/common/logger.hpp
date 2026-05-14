#pragma once
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

namespace bscs {

class Logger {
public:
    enum Level { DEBUG=0, INFO=1, WARN=2, ERR=3 };

    static Logger& get() { static Logger inst; return inst; }

    void set_level(Level l) { level_ = l; }

    template<typename... Args>
    void info(const char* fmt, Args&&... args) { log(INFO, "INFO ", fmt, args...); }

    template<typename... Args>
    void warn(const char* fmt, Args&&... args) { log(WARN, "WARN ", fmt, args...); }

    template<typename... Args>
    void err(const char* fmt, Args&&... args)  { log(ERR,  "ERR  ", fmt, args...); }

    template<typename... Args>
    void debug(const char* fmt, Args&&... args){ log(DEBUG,"DEBUG", fmt, args...); }

private:
    Level level_ = INFO;
    std::mutex mu_;

    template<typename... Args>
    void log(Level l, const char* tag, const char* fmt, Args&&... args) {
        if (l < level_) return;
        std::time_t t = std::time(nullptr);
        char ts[20];
        std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
        std::lock_guard<std::mutex> lk(mu_);
        std::fprintf(stderr, "[%s %s] ", ts, tag);
        std::fprintf(stderr, fmt, args...);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    }
};

#define LOG_INFO(...)  ::bscs::Logger::get().info(__VA_ARGS__)
#define LOG_WARN(...)  ::bscs::Logger::get().warn(__VA_ARGS__)
#define LOG_ERR(...)   ::bscs::Logger::get().err(__VA_ARGS__)
#define LOG_DEBUG(...) ::bscs::Logger::get().debug(__VA_ARGS__)

} // namespace bscs
