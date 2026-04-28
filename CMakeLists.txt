cmake_minimum_required(VERSION 3.16)
project(veilsight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VEILSIGHT_BUILD_TESTS "Build tests" ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTOBUF REQUIRED IMPORTED_TARGET protobuf)
pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)

find_program(PROTOC_EXECUTABLE protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin REQUIRED)

execute_process(
        COMMAND ${PROTOC_EXECUTABLE} --version
        OUTPUT_VARIABLE PROTOC_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE "^libprotoc ([0-9]+\\.[0-9]+).*" "\\1" PROTOC_VERSION_MAJOR_MINOR "${PROTOC_VERSION_OUTPUT}")
string(REGEX REPLACE "^([0-9]+\\.[0-9]+).*" "\\1" PROTOBUF_VERSION_MAJOR_MINOR "${PROTOBUF_VERSION}")
if (NOT PROTOC_VERSION_MAJOR_MINOR STREQUAL PROTOBUF_VERSION_MAJOR_MINOR)
    message(FATAL_ERROR
            "Protocol Buffers version mismatch: ${PROTOC_VERSION_OUTPUT}, "
            "pkg-config protobuf ${PROTOBUF_VERSION}. Install matching protoc, headers, and library.")
endif ()

set(VEILSIGHT_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(VEILSIGHT_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(VEILSIGHT_RUNNER_PROTO ${VEILSIGHT_PROTO_DIR}/veilsight/runner/v1/runner.proto)
set(VEILSIGHT_RUNNER_PROTO_SRCS
        ${VEILSIGHT_GENERATED_DIR}/veilsight/runner/v1/runner.pb.cc
        ${VEILSIGHT_GENERATED_DIR}/veilsight/runner/v1/runner.grpc.pb.cc
)
set(VEILSIGHT_RUNNER_PROTO_HDRS
        ${VEILSIGHT_GENERATED_DIR}/veilsight/runner/v1/runner.pb.h
        ${VEILSIGHT_GENERATED_DIR}/veilsight/runner/v1/runner.grpc.pb.h
)

add_custom_command(
        OUTPUT ${VEILSIGHT_RUNNER_PROTO_SRCS} ${VEILSIGHT_RUNNER_PROTO_HDRS}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${VEILSIGHT_GENERATED_DIR}
        COMMAND ${PROTOC_EXECUTABLE}
        ARGS --proto_path=${VEILSIGHT_PROTO_DIR}
             --cpp_out=${VEILSIGHT_GENERATED_DIR}
             --grpc_out=${VEILSIGHT_GENERATED_DIR}
             --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
             ${VEILSIGHT_RUNNER_PROTO}
        DEPENDS ${VEILSIGHT_RUNNER_PROTO}
        VERBATIM
)

add_library(veilsight_runner_proto ${VEILSIGHT_RUNNER_PROTO_SRCS} ${VEILSIGHT_RUNNER_PROTO_HDRS})
target_include_directories(veilsight_runner_proto PUBLIC ${VEILSIGHT_GENERATED_DIR})
target_link_libraries(veilsight_runner_proto PUBLIC PkgConfig::PROTOBUF PkgConfig::GRPCPP)

add_subdirectory(core)
add_subdirectory(apps)

if (VEILSIGHT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
