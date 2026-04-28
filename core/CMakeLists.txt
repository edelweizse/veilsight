cmake_minimum_required(VERSION 3.16)

add_library(veilsight_core)

# Headers
file(GLOB_RECURSE VEILSIGHT_CORE_HEADERS
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)

# Sources
file(GLOB_RECURSE VEILSIGHT_CORE_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
)
target_sources(veilsight_core PRIVATE ${VEILSIGHT_CORE_SOURCES} ${VEILSIGHT_CORE_HEADERS})
target_include_directories(veilsight_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty)

# Compile Options
target_compile_features(veilsight_core PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(veilsight_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Dependencies
# OpenCV
find_package(OpenCV QUIET)
target_link_libraries(veilsight_core PUBLIC ${OpenCV_LIBS})

# GStreamer
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_check_modules(GST_VIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)
pkg_check_modules(GST_ALLOCATORS REQUIRED IMPORTED_TARGET gstreamer-allocators-1.0)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)

target_link_libraries(veilsight_core PRIVATE
        PkgConfig::GST
        PkgConfig::GST_APP
        PkgConfig::GST_VIDEO
        PkgConfig::GST_ALLOCATORS
        PkgConfig::PCRE2
)

# YAML
find_package(yaml-cpp REQUIRED)
target_link_libraries(veilsight_core PRIVATE yaml-cpp::yaml-cpp)

# Threads
find_package(Threads REQUIRED)
target_link_libraries(veilsight_core PRIVATE Threads::Threads)

# NCNN
find_package(ncnn REQUIRED)
target_link_libraries(veilsight_core PRIVATE ncnn)
