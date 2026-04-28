cmake_minimum_required(VERSION 3.16)

add_executable(veilsight_core_service main.cpp)

target_sources(veilsight_core_service PRIVATE
        src/grpc_runner_server.cpp
        src/runner_manager.cpp
        src/telemetry_hub.cpp
)

target_include_directories(veilsight_core_service PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(veilsight_core_service PRIVATE
        veilsight_core
        veilsight_runner_proto
)
