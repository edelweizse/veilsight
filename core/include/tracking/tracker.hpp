#pragma once

#include <common/config.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <pipeline/types.hpp>

namespace veilsight {
    struct TrackerFrameInfo {
        std::string stream_id;
        int64_t frame_id = 0;
        int width = 0;
        int height = 0;
    };

    struct TrackerConfig {
        float high_thresh = 0.6f;
        float low_thresh = 0.2f;
        float match_iou_thresh = 0.3f;
        float low_match_iou_thresh = 0.2f;
        int min_hits = 2;
        int max_missed = 20;
    };

    class ITracker {
    public:
        virtual ~ITracker() = default;
        virtual std::vector<Box> update(const TrackerFrameInfo& frame,
                                        const std::vector<Box>& detections) = 0;
    };

    class ITrackerFactory {
    public:
        virtual ~ITrackerFactory() = default;
        virtual std::unique_ptr<ITracker> create() const = 0;
        virtual int backend_threads() const { return 1; }
    };

    std::unique_ptr<ITracker> create_demo_tracker(const TrackerConfig& cfg = {});
    std::unique_ptr<ITracker> create_bytetrack_tracker(const ByteTrackModuleConfig& cfg = {});
    std::unique_ptr<ITracker> create_ocsort_tracker(const OCSortModuleConfig& cfg = {});
    std::unique_ptr<ITrackerFactory> create_tracker_factory(const TrackerModuleConfig& cfg);
    std::unique_ptr<ITracker> create_tracker(const TrackerModuleConfig& cfg);
}
