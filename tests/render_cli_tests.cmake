if(NOT DEFINED RENDER_VIDEO)
    message(FATAL_ERROR "RENDER_VIDEO is required")
endif()
if(NOT DEFINED MODE)
    message(FATAL_ERROR "MODE is required")
endif()

set(output_path "${CMAKE_CURRENT_BINARY_DIR}/render_cli_${MODE}.mp4")

if(MODE STREQUAL "custom_requires_fps")
    execute_process(
            COMMAND "${RENDER_VIDEO}"
                    --config missing.yaml
                    --input missing.mp4
                    --output "${output_path}"
                    --timing-mode custom
            RESULT_VARIABLE result
            OUTPUT_VARIABLE stdout
            ERROR_VARIABLE stderr
    )
    string(CONCAT combined "${stdout}" "${stderr}")
    if(result EQUAL 0)
        message(FATAL_ERROR "custom timing without --fps unexpectedly succeeded")
    endif()
    if(NOT combined MATCHES "--fps > 0 is required when --timing-mode custom")
        message(FATAL_ERROR "custom timing failure did not mention missing --fps: ${combined}")
    endif()
elseif(MODE STREQUAL "source_without_fps")
    execute_process(
            COMMAND "${RENDER_VIDEO}"
                    --config missing.yaml
                    --input missing.mp4
                    --output "${output_path}"
                    --timing-mode source
                    --no-gallery
            RESULT_VARIABLE result
            OUTPUT_VARIABLE stdout
            ERROR_VARIABLE stderr
    )
    string(CONCAT combined "${stdout}" "${stderr}")
    if(result EQUAL 0)
        message(FATAL_ERROR "source timing with missing input unexpectedly succeeded")
    endif()
    if(combined MATCHES "Argument error")
        message(FATAL_ERROR "source timing without --fps failed during argument parsing: ${combined}")
    endif()
    if(NOT combined MATCHES "Cannot open input video")
        message(FATAL_ERROR "source timing failure did not reach input open: ${combined}")
    endif()
else()
    message(FATAL_ERROR "unknown MODE: ${MODE}")
endif()
