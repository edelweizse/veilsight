if (NOT DEFINED RENDER_VIDEO)
    message(FATAL_ERROR "RENDER_VIDEO is required")
endif ()

if (CASE STREQUAL "custom_requires_fps")
    execute_process(
            COMMAND "${RENDER_VIDEO}"
                    --config missing.yaml
                    --input missing.mp4
                    --output "${OUT}"
                    --timing-mode custom
            RESULT_VARIABLE result
            OUTPUT_VARIABLE stdout
            ERROR_VARIABLE stderr
    )
    if (result EQUAL 0)
        message(FATAL_ERROR "custom timing without --fps unexpectedly succeeded")
    endif ()
    if (NOT stderr MATCHES "--fps > 0 is required when --timing-mode custom")
        message(FATAL_ERROR "missing custom timing validation error: ${stderr}")
    endif ()
elseif (CASE STREQUAL "source_accepts_no_fps")
    execute_process(
            COMMAND "${RENDER_VIDEO}"
                    --config missing.yaml
                    --input missing.mp4
                    --output "${OUT}"
                    --timing-mode source
                    --no-gallery
            RESULT_VARIABLE result
            OUTPUT_VARIABLE stdout
            ERROR_VARIABLE stderr
    )
    if (result EQUAL 0)
        message(FATAL_ERROR "source timing with missing input unexpectedly succeeded")
    endif ()
    if (stderr MATCHES "Argument error")
        message(FATAL_ERROR "source timing without --fps failed during argument parsing: ${stderr}")
    endif ()
    if (NOT stderr MATCHES "Cannot open input video")
        message(FATAL_ERROR "source timing did not reach video open: ${stderr}")
    endif ()
elseif (CASE STREQUAL "mode_invalid")
    execute_process(
            COMMAND "${RENDER_VIDEO}"
                    --mode invalid
                    --config missing.yaml
                    --input missing.mp4
                    --output "${OUT}"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE stdout
            ERROR_VARIABLE stderr
    )
    if (result EQUAL 0)
        message(FATAL_ERROR "invalid mode unexpectedly succeeded")
    endif ()
    if (NOT stderr MATCHES "mode must be face or face\\+body")
        message(FATAL_ERROR "missing mode validation error: ${stderr}")
    endif ()
elseif (CASE STREQUAL "mode_face_parsed")
    execute_process(
            COMMAND "${RENDER_VIDEO}"
                    --mode face
                    --config missing.yaml
                    --input missing.mp4
                    --output "${OUT}"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE stdout
            ERROR_VARIABLE stderr
    )
    if (result EQUAL 0)
        message(FATAL_ERROR "face mode with missing config unexpectedly succeeded")
    endif ()
    if (stderr MATCHES "mode must be face")
        message(FATAL_ERROR "face mode rejected as invalid: ${stderr}")
    endif ()
elseif (CASE STREQUAL "mode_fb_default")
    execute_process(
            COMMAND "${RENDER_VIDEO}"
                    --config missing.yaml
                    --input missing.mp4
                    --output "${OUT}"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE stdout
            ERROR_VARIABLE stderr
    )
    if (result EQUAL 0)
        message(FATAL_ERROR "default mode with missing config unexpectedly succeeded")
    endif ()
    if (stderr MATCHES "mode must be face")
        message(FATAL_ERROR "default mode rejected as invalid: ${stderr}")
    endif ()
else ()
    message(FATAL_ERROR "unknown CASE: ${CASE}")
endif ()
