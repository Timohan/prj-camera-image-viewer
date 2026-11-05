#ifndef TG_GLOBAL_LOG_H
#define TG_GLOBAL_LOG_H

#include <iostream>
#include <iostream>
#include <string>
#include <cstdarg>

#ifndef TG_GLOBAL_LOG_LEVEL
#define TG_GLOBAL_LOG_LEVEL 1
#endif

#ifndef TG_GLOBAL_LOG_ENABLED
#define TG_GLOBAL_LOG_ENABLED
#endif

// writes the arguments to the stream separated by spaces, errors go to std::cerr and the others to std::cout
template <typename... Args>
inline void tgLog(std::ostream &stream, Args&&... args)
{
    std::string space;
    auto s = [&space, &stream](const auto& arg) -> decltype(arg) {
        if (space.empty()) {
            space = " ";
        } else {
            stream << space;
        }
        return arg;
    };
    (stream << ... << s(args)) << std::endl;
}

#ifdef TG_GLOBAL_LOG_ENABLED

#if TG_GLOBAL_LOG_LEVEL == 0
#define FUNCTION_BEGIN() \
    std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << " Begin" << std::endl;

#define FUNCTION_END() \
    std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << " End" << std::endl;
#else
#define TG_FUNCTION_BEGIN()
#define TG_FUNCTION_END()
#endif

#define TG_LOG(text) \
    std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << " " << text << std::endl;

#if TG_GLOBAL_LOG_LEVEL < 2
#define TG_DEBUG_LOG(...) tgLog(std::cout, "[D]", __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)
#else
#define TG_DEBUG_LOG(...)
#endif

#if TG_GLOBAL_LOG_LEVEL < 3
#define TG_INFO_LOG(...) tgLog(std::cout, "[I]", __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)
#else
#define TG_INFO_LOG(...)
#endif

#if TG_GLOBAL_LOG_LEVEL < 4
#define TG_WARNING_LOG(...) tgLog(std::cout, "[W]", __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)
#else
#define TG_WARNING_LOG(...)
#endif

#else
#define TG_FUNCTION_BEGIN()
#define TG_FUNCTION_END()
#define TG_LOG(text)
#define TG_DEBUG_LOG(...)
#define TG_INFO_LOG(...)
#define TG_WARNING_LOG(...)
#endif

#define TG_ERROR_LOG(...) tgLog(std::cerr, "[E]", __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)

#endif // TG_GLOBAL_LOG_H
