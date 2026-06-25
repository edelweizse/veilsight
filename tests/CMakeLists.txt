cmake_minimum_required(VERSION 3.16)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SQLITE3 REQUIRED IMPORTED_TARGET sqlite3)
find_package(ncnn REQUIRED)

add_executable(veilsight_config_tests config_tests.cpp)
target_link_libraries(veilsight_config_tests PRIVATE veilsight_core)
add_test(NAME veilsight_config_tests COMMAND veilsight_config_tests)

add_executable(veilsight_tracking_tests tracking_tests.cpp)
target_link_libraries(veilsight_tracking_tests PRIVATE veilsight_core)
add_test(NAME veilsight_tracking_tests COMMAND veilsight_tracking_tests)

add_executable(veilsight_face_tests face_tests.cpp)
target_link_libraries(veilsight_face_tests PRIVATE veilsight_core)
add_test(NAME veilsight_face_tests COMMAND veilsight_face_tests)

add_executable(veilsight_recognizer_tests recognizer_tests.cpp)
target_link_libraries(veilsight_recognizer_tests PRIVATE veilsight_core PkgConfig::SQLITE3 ncnn)
add_test(NAME veilsight_recognizer_tests COMMAND veilsight_recognizer_tests)

add_executable(veilsight_pipeline_tests pipeline_tests.cpp)
target_link_libraries(veilsight_pipeline_tests PRIVATE veilsight_core)
add_test(NAME veilsight_pipeline_tests COMMAND veilsight_pipeline_tests)

add_executable(veilsight_render_rules_tests render_rules_tests.cpp)
target_link_libraries(veilsight_render_rules_tests PRIVATE veilsight_core)
add_test(NAME veilsight_render_rules_tests COMMAND veilsight_render_rules_tests)

add_executable(veilsight_render_mode_tests render_mode_tests.cpp)
target_link_libraries(veilsight_render_mode_tests PRIVATE veilsight_core)
add_test(NAME veilsight_render_mode_tests COMMAND veilsight_render_mode_tests)

add_test(
        NAME veilsight_render_video_cli_custom_requires_fps
        COMMAND ${CMAKE_COMMAND}
                -DRENDER_VIDEO=$<TARGET_FILE:veilsight_render_video>
                -DCASE=custom_requires_fps
                -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_cli_custom_parse.mp4
                -P ${CMAKE_CURRENT_SOURCE_DIR}/render_video_cli_tests.cmake
)

add_test(
        NAME veilsight_render_video_cli_source_accepts_no_fps
        COMMAND ${CMAKE_COMMAND}
                -DRENDER_VIDEO=$<TARGET_FILE:veilsight_render_video>
                -DCASE=source_accepts_no_fps
                -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_cli_source_parse.mp4
                -P ${CMAKE_CURRENT_SOURCE_DIR}/render_video_cli_tests.cmake
)

add_test(
        NAME veilsight_render_video_cli_mode_invalid
        COMMAND ${CMAKE_COMMAND}
                -DRENDER_VIDEO=$<TARGET_FILE:veilsight_render_video>
                -DCASE=mode_invalid
                -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_cli_mode_invalid.mp4
                -P ${CMAKE_CURRENT_SOURCE_DIR}/render_video_cli_tests.cmake
)

add_test(
        NAME veilsight_render_video_cli_mode_face_parsed
        COMMAND ${CMAKE_COMMAND}
                -DRENDER_VIDEO=$<TARGET_FILE:veilsight_render_video>
                -DCASE=mode_face_parsed
                -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_cli_mode_face.mp4
                -P ${CMAKE_CURRENT_SOURCE_DIR}/render_video_cli_tests.cmake
)

add_test(
        NAME veilsight_render_video_cli_mode_fb_default
        COMMAND ${CMAKE_COMMAND}
                -DRENDER_VIDEO=$<TARGET_FILE:veilsight_render_video>
                -DCASE=mode_fb_default
                -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_cli_mode_fb.mp4
                -P ${CMAKE_CURRENT_SOURCE_DIR}/render_video_cli_tests.cmake
)
