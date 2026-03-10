#pragma once

#include <pipeline/types.hpp>
#include <tracking/tracker.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace ss {
    class StreamInferenceState {
    public:
        struct Callbacks {
            std::function<void(const FramePtr&)> on_frame_ready;
            std::function<void(const FrameCtx&, uint64_t)> on_tracker_timing;
        };

        StreamInferenceState(std::unique_ptr<ITracker> tracker,
                             int64_t reorder_window = 5,
                             size_t pending_limit = 500);

        void push_frame(const FramePtr& frame);
        void push_detection(InferResults det);
        void drain_ready(const Callbacks& callbacks);

    private:
        void process_frame_(const FramePtr& frame,
                            const std::vector<Box>& detections,
                            const Callbacks& callbacks);

        std::unique_ptr<ITracker> tracker_;
        int64_t next_frame_id_ = -1;
        int64_t reorder_window_ = 5;
        size_t pending_limit_ = 500;
        std::map<int64_t, FramePtr> pending_frames_;
        std::map<int64_t, InferResults> pending_detections_;
    };
}
