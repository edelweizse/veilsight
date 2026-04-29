#include <inference/detector.hpp>
#include <tracking/association.hpp>
#include <tracking/scene_grid.hpp>
#include <tracking/tracker.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace {
    int g_failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++g_failures;
            std::cerr << "[FAIL] " << message << "\n";
        }
    }

    veilsight::Box box(float x, float y, float w, float h, float score = 0.9f) {
        veilsight::Box b;
        b.x = x;
        b.y = y;
        b.w = w;
        b.h = h;
        b.score = score;
        return b;
    }

    veilsight::TrackerFrameInfo frame(int64_t frame_id) {
        return veilsight::TrackerFrameInfo{"cam0", frame_id, 640, 480};
    }

    void test_detector_factory_creates_yolox_factory() {
        veilsight::DetectorModuleConfig cfg;
        cfg.type = "yolox";
        cfg.yolox.ncnn_threads = 3;
        const auto factory = veilsight::create_detector_factory(cfg);
        check(factory != nullptr, "detector factory should create a YOLOX factory by detector.type");
        check(factory->backend_threads() == 3, "YOLOX factory should expose configured NCNN internal threads");
    }

    void test_yolox_ncnn_detector_loads_and_runs() {
        veilsight::DetectorModuleConfig cfg;
        cfg.type = "yolox";
        cfg.yolox.variant = "nano";
        cfg.yolox.input_w = 1088;
        cfg.yolox.input_h = 608;
        cfg.yolox.decoded_output = false;

        auto detector = veilsight::create_detector(cfg);
        const cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
        const auto boxes = detector->detect(frame);
        for (const auto& b : boxes) {
            check(b.x >= 0.0f && b.y >= 0.0f, "YOLOX boxes should be clipped to frame origin");
            check(b.x + b.w <= 320.1f && b.y + b.h <= 240.1f, "YOLOX boxes should be clipped to frame bounds");
            check(b.id == -1, "YOLOX detections should not assign track IDs");
        }
    }

    void test_association_returns_deterministic_matches() {
        const std::vector<veilsight::Box> tracks = {
            box(10, 10, 100, 200),
            box(300, 10, 100, 200),
        };
        const std::vector<veilsight::Box> detections = {
            box(302, 12, 100, 200),
            box(12, 12, 100, 200),
        };

        const auto result = veilsight::associate_detections(
            tracks,
            detections,
            veilsight::AssociationOptions{
                veilsight::AssociationStage::High,
                0.5f,
                false,
                nullptr,
                "cam0",
                640,
                480,
            });

        check(result.matches.size() == 2, "association should match both tracks");
        check(result.matches[0].track_index == 0 && result.matches[0].detection_index == 1,
              "association should match first track to first spatial detection");
        check(result.matches[1].track_index == 1 && result.matches[1].detection_index == 0,
              "association should match second track to second spatial detection");
        check(result.unmatched_tracks.empty(), "association should leave no unmatched tracks");
        check(result.unmatched_detections.empty(), "association should leave no unmatched detections");
    }

    void test_grid_disabled_equals_baseline_association() {
        const std::vector<veilsight::Box> tracks = {box(10, 10, 100, 200)};
        const std::vector<veilsight::Box> detections = {box(15, 10, 100, 200)};

        veilsight::SceneGridConfig grid_cfg;
        grid_cfg.enabled = false;
        veilsight::SceneGrid grid(grid_cfg);
        grid.begin_frame("cam0");

        const auto baseline = veilsight::associate_detections(
            tracks,
            detections,
            veilsight::AssociationOptions{veilsight::AssociationStage::High, 0.5f, false, nullptr, "cam0", 640, 480});
        const auto guided = veilsight::associate_detections(
            tracks,
            detections,
            veilsight::AssociationOptions{veilsight::AssociationStage::High, 0.5f, false, &grid, "cam0", 640, 480});

        check(baseline.matches.size() == 1 && guided.matches.size() == 1,
              "baseline and disabled grid association should both match");
        check(std::fabs(baseline.matches[0].total_cost - guided.matches[0].total_cost) < 0.0001f,
              "disabled grid should not alter association cost");
    }

    void test_grid_soft_cost_changes_ambiguous_association() {
        veilsight::SceneGridConfig grid_cfg;
        grid_cfg.rows = 2;
        grid_cfg.cols = 2;
        grid_cfg.warmup_frames = 0;
        grid_cfg.association_weight = 0.30f;
        grid_cfg.max_extra_cost = 0.30f;
        grid_cfg.cell_distance_weight = 1.0f;
        grid_cfg.occupancy_weight = 0.0f;
        grid_cfg.transition_weight = 0.0f;
        veilsight::SceneGrid grid(grid_cfg);
        grid.begin_frame("cam0");

        const std::vector<veilsight::Box> tracks = {box(0, 140, 100, 100)};
        const std::vector<veilsight::Box> detections = {
            box(8, 140, 100, 100),
            box(0, 145, 100, 100),
        };

        const auto baseline = veilsight::associate_detections(
            tracks,
            detections,
            veilsight::AssociationOptions{veilsight::AssociationStage::High, 0.0f, false, nullptr, "cam0", 640, 480});
        const auto guided = veilsight::associate_detections(
            tracks,
            detections,
            veilsight::AssociationOptions{veilsight::AssociationStage::High, 0.0f, false, &grid, "cam0", 640, 480});

        check(baseline.matches.size() == 1 && baseline.matches[0].detection_index == 1,
              "baseline should choose the slightly higher IoU detection");
        check(guided.matches.size() == 1 && guided.matches[0].detection_index == 0,
              "grid should change an ambiguous association toward the same floor cell");
    }

    void test_grid_cost_cannot_force_match_past_threshold() {
        veilsight::SceneGridConfig grid_cfg;
        grid_cfg.warmup_frames = 0;
        grid_cfg.association_weight = 0.0f;
        veilsight::SceneGrid grid(grid_cfg);
        grid.begin_frame("cam0");

        const auto result = veilsight::associate_detections(
            {box(0, 0, 100, 100)},
            {box(300, 0, 100, 100)},
            veilsight::AssociationOptions{veilsight::AssociationStage::High, 0.5f, false, &grid, "cam0", 640, 480});

        check(result.matches.empty(), "grid guidance should not force a match below IoU threshold");
        check(result.unmatched_tracks.size() == 1, "low-IoU track should remain unmatched");
        check(result.unmatched_detections.size() == 1, "low-IoU detection should remain unmatched");
    }

    void test_score_fusion_does_not_raise_iou_threshold() {
        const auto result = veilsight::associate_detections(
            {box(0, 0, 100, 100)},
            {box(10, 0, 100, 100, 0.5f)},
            veilsight::AssociationOptions{veilsight::AssociationStage::High, 0.7f, true, nullptr, "cam0", 640, 480});

        check(result.matches.size() == 1, "score fusion should not reject a match above the raw IoU threshold");
        check(result.unmatched_tracks.empty(), "score-fused valid IoU match should leave no unmatched track");
        check(result.unmatched_detections.empty(), "score-fused valid IoU match should leave no unmatched detection");
    }

    void test_grid_cost_does_not_raise_iou_threshold() {
        veilsight::SceneGridConfig grid_cfg;
        grid_cfg.warmup_frames = 0;
        grid_cfg.association_weight = 0.30f;
        grid_cfg.occupancy_weight = 1.0f;
        grid_cfg.transition_weight = 1.0f;
        veilsight::SceneGrid grid(grid_cfg);
        grid.begin_frame("cam0");

        const auto result = veilsight::associate_detections(
            {box(0, 0, 100, 100)},
            {box(5, 0, 100, 100)},
            veilsight::AssociationOptions{veilsight::AssociationStage::High, 0.9f, false, &grid, "cam0", 640, 480});

        check(result.matches.size() == 1, "scene grid cost should not reject a match above the raw IoU threshold");
        check(result.unmatched_tracks.empty(), "grid-guided valid IoU match should leave no unmatched track");
        check(result.unmatched_detections.empty(), "grid-guided valid IoU match should leave no unmatched detection");
    }

    void test_bottom_center_cell_mapping_clamps_edges() {
        veilsight::SceneGridConfig cfg;
        cfg.rows = 2;
        cfg.cols = 2;
        veilsight::SceneGrid grid(cfg);

        const auto top_left = grid.cell_for_box(100, 100, box(-20, -20, 10, 10));
        const auto bottom_right = grid.cell_for_box(100, 100, box(95, 95, 20, 20));

        check(top_left.first == 0 && top_left.second == 0, "grid cell should clamp top-left edge");
        check(bottom_right.first == 1 && bottom_right.second == 1, "grid cell should clamp bottom-right edge");
    }

    void test_bytetrack_preserves_id_after_short_miss() {
        veilsight::ByteTrackModuleConfig cfg;
        cfg.high_thresh = 0.6f;
        cfg.low_thresh = 0.1f;
        cfg.new_track_thresh = 0.6f;
        cfg.match_iou_thresh = 0.5f;
        cfg.track_buffer = 3;
        cfg.min_box_area = 100.0f;
        cfg.fuse_score = false;
        cfg.scene_grid.enabled = false;
        auto tracker = veilsight::create_bytetrack_tracker(cfg);

        const auto out1 = tracker->update(frame(1), {box(100, 50, 80, 180, 0.9f)});
        const auto out2 = tracker->update(frame(2), {});
        const auto out3 = tracker->update(frame(3), {box(105, 50, 80, 180, 0.9f)});

        check(out1.size() == 1, "ByteTrack should emit initial person track");
        check(out2.empty(), "ByteTrack should not emit lost tracks by default");
        check(out3.size() == 1, "ByteTrack should re-emit matched track after short miss");
        check(!out1.empty() && !out3.empty() && out1[0].id == out3[0].id,
              "ByteTrack should preserve person track ID after short miss");
    }

    void test_bytetrack_uses_low_score_detections_with_fuse_score() {
        veilsight::ByteTrackModuleConfig cfg;
        cfg.high_thresh = 0.6f;
        cfg.low_thresh = 0.2f;
        cfg.new_track_thresh = 0.6f;
        cfg.match_iou_thresh = 0.35f;
        cfg.low_match_iou_thresh = 0.25f;
        cfg.min_box_area = 0.0f;
        cfg.fuse_score = true;
        cfg.scene_grid.enabled = false;
        auto tracker = veilsight::create_bytetrack_tracker(cfg);

        const auto out1 = tracker->update(frame(1), {box(100, 50, 80, 180, 0.9f)});
        const auto out2 = tracker->update(frame(2), {box(108, 50, 80, 180, 0.25f)});

        check(out1.size() == 1, "ByteTrack should start from a high-confidence detection");
        check(out2.size() == 1, "ByteTrack should recover with a low-score detection when IoU is valid");
        check(!out1.empty() && !out2.empty() && out1[0].id == out2[0].id,
              "ByteTrack low-score recovery should preserve the existing track ID");
    }

    void test_bytetrack_allows_person_scale_change_without_face_clamp() {
        veilsight::ByteTrackModuleConfig cfg;
        cfg.high_thresh = 0.6f;
        cfg.low_thresh = 0.1f;
        cfg.new_track_thresh = 0.6f;
        cfg.match_iou_thresh = 0.1f;
        cfg.min_box_area = 100.0f;
        cfg.fuse_score = false;
        cfg.scene_grid.enabled = false;
        auto tracker = veilsight::create_bytetrack_tracker(cfg);

        (void)tracker->update(frame(1), {box(100, 50, 80, 180, 0.9f)});
        const auto out = tracker->update(frame(2), {box(90, 20, 300, 300, 0.9f)});

        check(out.size() == 1, "ByteTrack should keep matching a valid larger person box");
        check(!out.empty() && out[0].w > 116.0f,
              "ByteTrack should not clamp full-body width to the old face-oriented 1.45x limit");
    }
}

int main() {
    test_detector_factory_creates_yolox_factory();
    test_yolox_ncnn_detector_loads_and_runs();
    test_association_returns_deterministic_matches();
    test_grid_disabled_equals_baseline_association();
    test_grid_soft_cost_changes_ambiguous_association();
    test_grid_cost_cannot_force_match_past_threshold();
    test_score_fusion_does_not_raise_iou_threshold();
    test_grid_cost_does_not_raise_iou_threshold();
    test_bottom_center_cell_mapping_clamps_edges();
    test_bytetrack_preserves_id_after_short_miss();
    test_bytetrack_uses_low_score_detections_with_fuse_score();
    test_bytetrack_allows_person_scale_change_without_face_clamp();

    if (g_failures != 0) {
        std::cerr << "[FAIL] total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "[OK] all tracking tests passed\n";
    return 0;
}
