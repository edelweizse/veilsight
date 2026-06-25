#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <pipeline/types.hpp>

namespace veilsight {
    struct AnonymizerConfig {
        // Supported methods: "pixelate", "blur"
        std::string method = "pixelate";

        // Pixelate: downscale ROI by this factor, then upsample with nearest-neighbor.
        int pixelation_divisor = 10;

        // Blur: gaussian kernel size (will be forced to odd and >= 3).
        int blur_kernel = 31;

        // When true, anonymize the detected face ROI for anonymized tracks and
        // fall back to the full person box when no face is attached.
        bool face_only_when_available = false;

        // When true (and face_only_when_available is also true), boxes without
        // face data are skipped entirely — no body ROI fallback.
        // Used by face-only rendering mode.
        bool strict_face_only = false;
    };

    struct AnonymizationRegion {
        cv::Rect roi;
        int source_track_id = -1;
        std::string source_track_type = "body";
        std::string target_type = "body";
        std::string method;
    };

    class Anonymizer {
    public:
        explicit Anonymizer(AnonymizerConfig cfg);

        std::vector<AnonymizationRegion> planned_regions(
            const cv::Mat& ui_frame,
            const std::vector<Box>& boxes_inf_space,
            float sx,
            float sy,
            float tx,
            float ty) const;

        void apply(cv::Mat& ui_frame,
                   const std::vector<Box>& boxes_inf_space,
                   float sx,
                   float sy,
                   float tx,
                   float ty) const;

    private:
        enum class Method {
            Pixelate,
            Blur
        };

        cv::Rect map_box_to_ui_(const Box& b,
                                float sx,
                                float sy,
                                float tx,
                                float ty,
                                int ui_w,
                                int ui_h) const;

        void apply_pixelate_(cv::Mat& roi) const;
        void apply_blur_(cv::Mat& roi) const;

        Method method_ = Method::Pixelate;
        int pixelation_divisor_ = 10;
        int blur_kernel_ = 31;
        bool face_only_when_available_ = false;
        bool strict_face_only_ = false;
    };
}
