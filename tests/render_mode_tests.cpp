#include <anonymization/anonymizer.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {
    int g_failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++g_failures;
            std::cerr << "[FAIL] " << message << "\n";
        }
    }

    veilsight::Box body_box(int id = 1) {
        veilsight::Box box;
        box.id = id;
        box.x = 50.0f;
        box.y = 100.0f;
        box.w = 200.0f;
        box.h = 300.0f;
        box.score = 0.9f;
        box.privacy_action = "anonymize";
        return box;
    }

    void test_strict_face_only_skips_body_box() {
        veilsight::AnonymizerConfig cfg;
        cfg.face_only_when_available = true;
        cfg.strict_face_only = true;
        cfg.method = "pixelate";
        veilsight::Anonymizer anon(cfg);

        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
        std::vector<veilsight::Box> boxes = {body_box(1)};
        auto regions = anon.planned_regions(frame, boxes, 1.0f, 1.0f, 0.0f, 0.0f);

        check(regions.empty(),
              "strict face-only should skip box without face data");
    }

    void test_strict_face_only_keeps_face_box() {
        veilsight::AnonymizerConfig cfg;
        cfg.face_only_when_available = true;
        cfg.strict_face_only = true;
        cfg.method = "pixelate";
        veilsight::Anonymizer anon(cfg);

        veilsight::Box box = body_box(1);
        box.face = veilsight::FaceObservation{};
        box.face->bbox = {50.0f, 100.0f, 80.0f, 60.0f};
        box.face->score = 0.9f;

        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
        auto regions = anon.planned_regions(frame, {box}, 1.0f, 1.0f, 0.0f, 0.0f);

        check(regions.size() == 1,
              "strict face-only should keep box with face data");
        check(regions[0].target_type == "face",
              "strict face-only should use face target type: expected 'face', got '" + regions[0].target_type + "'");
    }

    void test_non_strict_preserves_body_fallback() {
        veilsight::AnonymizerConfig cfg;
        cfg.face_only_when_available = true;
        cfg.strict_face_only = false;
        cfg.method = "pixelate";
        veilsight::Anonymizer anon(cfg);

        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
        auto regions = anon.planned_regions(frame, {body_box(1)}, 1.0f, 1.0f, 0.0f, 0.0f);

        check(regions.size() == 1,
              "non-strict should fall back to body when face data missing");
        check(regions[0].target_type == "body",
              "non-strict should use body target type when no face data");
    }

    void test_strict_face_only_requires_face_only_when_available() {
        veilsight::AnonymizerConfig cfg;
        cfg.face_only_when_available = false;
        cfg.strict_face_only = true;
        cfg.method = "pixelate";
        veilsight::Anonymizer anon(cfg);

        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
        auto regions = anon.planned_regions(frame, {body_box(1)}, 1.0f, 1.0f, 0.0f, 0.0f);

        check(regions.size() == 1,
              "strict_face_only without face_only_when_available should not skip boxes");
    }

    void test_anonymize_applies_blur() {
        veilsight::AnonymizerConfig cfg;
        cfg.face_only_when_available = false;
        cfg.strict_face_only = false;
        cfg.method = "blur";
        cfg.blur_kernel = 31;
        veilsight::Anonymizer anon(cfg);

        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
        std::vector<veilsight::Box> boxes = {body_box(1)};
        anon.apply(frame, boxes, 1.0f, 1.0f, 0.0f, 0.0f);
        check(true, "anonymize apply (blur) should not crash");
    }
}

int main() {
    test_strict_face_only_skips_body_box();
    test_strict_face_only_keeps_face_box();
    test_non_strict_preserves_body_fallback();
    test_strict_face_only_requires_face_only_when_available();
    test_anonymize_applies_blur();

    if (g_failures) {
        std::cerr << g_failures << " render mode test(s) failed\n";
        return 1;
    }
    std::cout << "render mode tests passed\n";
    return 0;
}
