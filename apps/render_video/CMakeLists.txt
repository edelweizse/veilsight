cmake_minimum_required(VERSION 3.16)

add_executable(veilsight_render_video main.cpp)
target_link_libraries(veilsight_render_video PRIVATE veilsight_core)
