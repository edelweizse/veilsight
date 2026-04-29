#include <inference/stream_inference_state.hpp>

#include <algorithm>
#include <chrono>
#include <utility>

namespace veilsight {
    StreamInferenceState::StreamInferenceState(std::unique_ptr<ITracker> tracker,
                                               int64_t reorder_window,
                                               size_t pending_limit)
        : tracker_(std::move(tracker)),
          reorder_window_(std::max<int64_t>(0, reorder_window)),
          pending_limit_(std::max<size_t>(1, pending_limit)) {}

    void StreamInferenceState::push_frame(const FramePtr& frame) {
        if (!frame) return;
        pending_frames_[frame->frame_id] = frame;
    }

    void StreamInferenceState::push_detection(InferResults det) {
        pending_detections_[det.frame_id] = std::move(det);
    }

    void StreamInferenceState::drain_ready(const Callbacks& callbacks) {
        if (next_frame_id_ < 0 && !pending_frames_.empty()) {
            next_frame_id_ = pending_frames_.begin()->first;
        }

        while (next_frame_id_ >= 0) {
            auto frame_it = pending_frames_.find(next_frame_id_);
            auto det_it = pending_detections_.find(next_frame_id_);

            if (frame_it != pending_frames_.end() && det_it != pending_detections_.end()) {
                process_frame_(frame_it->second, det_it->second.bboxes, callbacks);
                pending_frames_.erase(frame_it);
                pending_detections_.erase(det_it);
                ++next_frame_id_;
                continue;
            }

            if (frame_it == pending_frames_.end()) {
                if (!pending_frames_.empty() && pending_frames_.begin()->first > next_frame_id_) {
                    next_frame_id_ = pending_frames_.begin()->first;
                    continue;
                }
                break;
            }

            const int64_t latest_frame = pending_frames_.empty() ? next_frame_id_ : pending_frames_.rbegin()->first;
            const int64_t latest_det = pending_detections_.empty() ? next_frame_id_ : pending_detections_.rbegin()->first;
            const int64_t latest_seen = std::max(latest_frame, latest_det);
            if (latest_seen - next_frame_id_ > reorder_window_) {
                process_frame_(frame_it->second, {}, callbacks);
                pending_frames_.erase(frame_it);
                ++next_frame_id_;
                continue;
            }
            break;
        }

        while (pending_frames_.size() > pending_limit_) {
            pending_frames_.erase(pending_frames_.begin());
        }
        while (pending_detections_.size() > pending_limit_) {
            pending_detections_.erase(pending_detections_.begin());
        }
    }

    void StreamInferenceState::process_frame_(const FramePtr& frame,
                                              const std::vector<Box>& detections,
                                              const Callbacks& callbacks) {
        if (!frame || !tracker_) return;

        const auto tracker_t0 = std::chrono::steady_clock::now();
        frame->tracked_boxes = tracker_->update(
            TrackerFrameInfo{
                frame->stream_id,
                frame->frame_id,
                frame->inf_w,
                frame->inf_h,
            },
            detections);
        const auto tracker_t1 = std::chrono::steady_clock::now();

        if (callbacks.on_tracker_timing) {
            callbacks.on_tracker_timing(
                *frame,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(tracker_t1 - tracker_t0).count()));
        }
        if (callbacks.on_frame_ready) {
            callbacks.on_frame_ready(frame);
        }
    }
}
