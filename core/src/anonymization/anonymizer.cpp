#include <anonymization/anonymizer.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace veilsight {
    namespace {
        std::string normalize_method(std::string s) {
            std::transform(s.begin(),
                           s.end(),
                           s.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }
    } // namespace

    Anonymizer::Anonymizer(AnonymizerConfig cfg) {
        const std::string method = normalize_method(std::move(cfg.method));
        if (method == "blur") {
            method_ = Method::Blur;
        } else {
            method_ = Method::Pixelate;
        }

        pixelation_divisor_ = std::max(2, cfg.pixelation_divisor);
        blur_kernel_ = std::max(3, cfg.blur_kernel);
        if ((blur_kernel_ % 2) == 0) blur_kernel_ += 1;
        face_only_when_available_ = cfg.face_only_when_available;
        strict_face_only_ = cfg.strict_face_only;
    }

    std::vector<AnonymizationRegion> Anonymizer::planned_regions(
        const cv::Mat& ui_frame,
        const std::vector<Box>& boxes_inf_space,
        float sx,
        float sy,
        float tx,
        float ty) const {
        std::vector<AnonymizationRegion> regions;
        if (ui_frame.empty()) return regions;

        for (const auto& b : boxes_inf_space) {
            if (b.privacy_action != "anonymize") continue;
            if (b.w <= 1.0f || b.h <= 1.0f) continue;

            if (strict_face_only_ && face_only_when_available_ &&
                (!b.face.has_value() || b.face->bbox.w <= 1.0f || b.face->bbox.h <= 1.0f)) {
                continue;
            }

            Box roi_box = b;
            bool using_face_roi = false;
            if (face_only_when_available_ && b.face.has_value() &&
                b.face->bbox.w > 1.0f && b.face->bbox.h > 1.0f) {
                roi_box.x = b.face->bbox.x;
                roi_box.y = b.face->bbox.y;
                roi_box.w = b.face->bbox.w;
                roi_box.h = b.face->bbox.h;
                roi_box.occluded = false;
                using_face_roi = true;
            }

            cv::Rect roi_rect =
                map_box_to_ui_(roi_box, sx, sy, tx, ty, ui_frame.cols, ui_frame.rows);
            if (using_face_roi && (roi_rect.width < 2 || roi_rect.height < 2)) {
                roi_rect = map_box_to_ui_(b, sx, sy, tx, ty, ui_frame.cols, ui_frame.rows);
                using_face_roi = false;
            }
            if (roi_rect.width < 2 || roi_rect.height < 2) continue;

            AnonymizationRegion region;
            region.roi = roi_rect;
            region.source_track_id = b.id;
            region.source_track_type = b.id < 0 ? "face" : "body";
            region.target_type = using_face_roi ? "face" : "body";
            region.method = method_ == Method::Blur ? "blur" : "pixelate";
            regions.push_back(std::move(region));
        }
        return regions;
    }

    void Anonymizer::apply(cv::Mat& ui_frame,
                           const std::vector<Box>& boxes_inf_space,
                           float sx,
                           float sy,
                           float tx,
                           float ty) const {
        for (const auto& region : planned_regions(ui_frame, boxes_inf_space, sx, sy, tx, ty)) {
            cv::Mat roi = ui_frame(region.roi);
            if (method_ == Method::Blur) {
                apply_blur_(roi);
            } else {
                apply_pixelate_(roi);
            }
        }
    }

    cv::Rect Anonymizer::map_box_to_ui_(const Box& b,
                                        float sx,
                                        float sy,
                                        float tx,
                                        float ty,
                                        int ui_w,
                                        int ui_h) const {
        // Expand regions so anonymization remains fail-closed under partial occlusion.
        const float inflate = b.occluded ? 1.55f : 1.30f;
        const float cx = b.x + b.w * 0.5f;
        const float cy = b.y + b.h * 0.5f;
        const float ew = std::max(1.0f, b.w * inflate);
        const float eh = std::max(1.0f, b.h * inflate);
        const float ex = cx - ew * 0.5f;
        const float ey = cy - eh * 0.5f;

        const int x = static_cast<int>(std::lround(ex * sx + tx));
        const int y = static_cast<int>(std::lround(ey * sy + ty));
        const int w = static_cast<int>(std::lround(ew * sx));
        const int h = static_cast<int>(std::lround(eh * sy));

        cv::Rect r(x, y, w, h);
        cv::Rect bounds(0, 0, ui_w, ui_h);
        r &= bounds;
        return r;
    }

    void Anonymizer::apply_pixelate_(cv::Mat& roi) const {
        cv::Mat tiny;
        const int tw = std::max(2, roi.cols / pixelation_divisor_);
        const int th = std::max(2, roi.rows / pixelation_divisor_);
        cv::resize(roi, tiny, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);
        cv::resize(tiny, roi, roi.size(), 0, 0, cv::INTER_NEAREST);
    }

    void Anonymizer::apply_blur_(cv::Mat& roi) const {
        cv::GaussianBlur(roi, roi, cv::Size(blur_kernel_, blur_kernel_), 0.0, 0.0);
    }
}
