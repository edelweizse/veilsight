cmake_minimum_required(VERSION 3.16)
add_executable(enroll_faces main.cpp)
target_link_libraries(enroll_faces PRIVATE veilsight_core)
