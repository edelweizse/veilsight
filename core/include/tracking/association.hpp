#pragma once

#include <pipeline/types.hpp>
#include <tracking/scene_grid.hpp>

#include <string>
#include <vector>

namespace veilsight {
    enum class AssociationStage {
        High,
        Low,
        Unconfirmed
    };

    struct AssociationMatch {
        int track_index = -1;
        int detection_index = -1;
        float total_cost = 1.0f;
        float iou = 0.0f;
        float grid_cost = 0.0f;
        AssociationStage stage = AssociationStage::High;
    };

    struct AssociationResult {
        std::vector<AssociationMatch> matches;
        std::vector<int> unmatched_tracks;
        std::vector<int> unmatched_detections;
    };

    struct AssociationOptions {
        AssociationStage stage = AssociationStage::High;
        float iou_threshold = 0.5f;
        bool fuse_score = true;
        const SceneGrid* scene_grid = nullptr;
        std::string stream_id;
        int frame_width = 0;
        int frame_height = 0;
    };

    float box_iou(const Box& a, const Box& b);
    std::vector<int> hungarian_assignment(const std::vector<std::vector<float>>& cost);
    AssociationResult associate_detections(const std::vector<Box>& tracks,
                                           const std::vector<Box>& detections,
                                           const AssociationOptions& options);
}
