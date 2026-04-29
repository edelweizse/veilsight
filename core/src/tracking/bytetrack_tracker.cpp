#include <tracking/tracker.hpp>
#include <tracking/association.hpp>
#include <tracking/scene_grid.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace veilsight {
namespace {

enum class TrackState : uint8_t {
    Tracked = 0,
    Lost = 1,
    Removed = 2,
};

struct Track {
    int id = 0;
    int64_t start_frame = 0;
    int64_t frame_id = 0;
    int tracklet_len = 0;
    float score = 0.0f;
    bool is_activated = false;
    TrackState state = TrackState::Tracked;
    Box tlwh{};

    cv::Mat mean;       // 8x1 (x, y, a, h, vx, vy, va, vh)
    cv::Mat covariance; // 8x8

    int64_t end_frame() const { return frame_id; }
};

using TrackPtr = std::shared_ptr<Track>;

class ByteKalmanFilter {
public:
    ByteKalmanFilter() {
        motion_mat_ = cv::Mat::eye(8, 8, CV_32F);
        for (int i = 0; i < 4; ++i) motion_mat_.at<float>(i, i + 4) = 1.0f;

        update_mat_ = cv::Mat::zeros(4, 8, CV_32F);
        for (int i = 0; i < 4; ++i) update_mat_.at<float>(i, i) = 1.0f;
    }

    void initiate(const cv::Vec4f& measurement, cv::Mat& mean, cv::Mat& covariance) const {
        mean = cv::Mat::zeros(8, 1, CV_32F);
        mean.at<float>(0, 0) = measurement[0];
        mean.at<float>(1, 0) = measurement[1];
        mean.at<float>(2, 0) = measurement[2];
        mean.at<float>(3, 0) = measurement[3];

        const float h = std::max(1.0f, measurement[3]);
        const std::vector<float> std = {
            2.0f * std_weight_position_ * h,
            2.0f * std_weight_position_ * h,
            1e-2f,
            2.0f * std_weight_position_ * h,
            10.0f * std_weight_velocity_ * h,
            10.0f * std_weight_velocity_ * h,
            1e-5f,
            10.0f * std_weight_velocity_ * h,
        };

        covariance = cv::Mat::zeros(8, 8, CV_32F);
        for (int i = 0; i < 8; ++i) covariance.at<float>(i, i) = std[i] * std[i];
    }

    void predict(cv::Mat& mean, cv::Mat& covariance) const {
        if (mean.empty() || covariance.empty()) return;

        const float h = std::max(1.0f, mean.at<float>(3, 0));
        const std::vector<float> std = {
            std_weight_position_ * h,
            std_weight_position_ * h,
            1e-2f,
            std_weight_position_ * h,
            std_weight_velocity_ * h,
            std_weight_velocity_ * h,
            1e-5f,
            std_weight_velocity_ * h,
        };

        cv::Mat motion_cov = cv::Mat::zeros(8, 8, CV_32F);
        for (int i = 0; i < 8; ++i) motion_cov.at<float>(i, i) = std[i] * std[i];

        mean = motion_mat_ * mean;
        covariance = motion_mat_ * covariance * motion_mat_.t() + motion_cov;
    }

    void update(cv::Mat& mean, cv::Mat& covariance, const cv::Vec4f& measurement) const {
        if (mean.empty() || covariance.empty()) return;

        cv::Mat projected_mean;
        cv::Mat projected_covariance;
        project_(mean, covariance, projected_mean, projected_covariance);

        cv::Mat innovation = cv::Mat::zeros(4, 1, CV_32F);
        innovation.at<float>(0, 0) = measurement[0] - projected_mean.at<float>(0, 0);
        innovation.at<float>(1, 0) = measurement[1] - projected_mean.at<float>(1, 0);
        innovation.at<float>(2, 0) = measurement[2] - projected_mean.at<float>(2, 0);
        innovation.at<float>(3, 0) = measurement[3] - projected_mean.at<float>(3, 0);

        cv::Mat projected_cov_inv;
        cv::invert(projected_covariance, projected_cov_inv, cv::DECOMP_SVD);
        const cv::Mat kalman_gain = covariance * update_mat_.t() * projected_cov_inv;

        mean = mean + kalman_gain * innovation;
        covariance = covariance - kalman_gain * projected_covariance * kalman_gain.t();
    }

private:
    void project_(const cv::Mat& mean,
                  const cv::Mat& covariance,
                  cv::Mat& projected_mean,
                  cv::Mat& projected_covariance) const {
        const float h = std::max(1.0f, mean.at<float>(3, 0));
        const std::vector<float> std = {
            std_weight_position_ * h,
            std_weight_position_ * h,
            1e-1f,
            std_weight_position_ * h,
        };

        cv::Mat innovation_cov = cv::Mat::zeros(4, 4, CV_32F);
        for (int i = 0; i < 4; ++i) innovation_cov.at<float>(i, i) = std[i] * std[i];

        projected_mean = update_mat_ * mean;
        projected_covariance = update_mat_ * covariance * update_mat_.t() + innovation_cov;
    }

    cv::Mat motion_mat_;
    cv::Mat update_mat_;
    const float std_weight_position_ = 1.0f / 20.0f;
    const float std_weight_velocity_ = 1.0f / 160.0f;
};

// ---------------------------
// Geometry helpers
// ---------------------------
static inline float area_of(const Box& b) {
    return std::max(0.0f, b.w) * std::max(0.0f, b.h);
}

static inline cv::Vec4f tlwh_to_xyah(const Box& b) {
    const float w = std::max(1.0f, b.w);
    const float h = std::max(1.0f, b.h);
    return cv::Vec4f(b.x + w * 0.5f, b.y + h * 0.5f, w / h, h);
}

static inline Box xyah_to_tlwh(const cv::Mat& mean) {
    const float cx = mean.at<float>(0, 0);
    const float cy = mean.at<float>(1, 0);
    const float a  = mean.at<float>(2, 0);
    const float h  = std::max(1.0f, mean.at<float>(3, 0));
    const float w  = std::max(1.0f, a * h);

    Box out;
    out.x = cx - w * 0.5f;
    out.y = cy - h * 0.5f;
    out.w = w;
    out.h = h;
    return out;
}

static inline void damp_or_reset_velocity(cv::Mat& mean, TrackState state) {
    if (mean.empty()) return;

    if (state != TrackState::Tracked) {
        for (int i = 4; i < 8; ++i) mean.at<float>(i, 0) = 0.0f;
    }
}

static inline void predict_track(Track& track, const ByteKalmanFilter& kf) {
    if (track.mean.empty() || track.covariance.empty()) return;

    damp_or_reset_velocity(track.mean, track.state);
    kf.predict(track.mean, track.covariance);
    track.tlwh = xyah_to_tlwh(track.mean);
}

static inline void activate_track(Track& track,
                                  const ByteKalmanFilter& kf,
                                  int64_t frame_id,
                                  int track_id) {
    const cv::Vec4f xyah = tlwh_to_xyah(track.tlwh);
    kf.initiate(xyah, track.mean, track.covariance);
    track.tlwh = xyah_to_tlwh(track.mean);

    track.id = track_id;
    track.frame_id = frame_id;
    track.start_frame = frame_id;
    track.tracklet_len = 0;
    track.state = TrackState::Tracked;

    track.is_activated = true;
}

static inline void update_track(Track& track,
                                const Track& detection,
                                const ByteKalmanFilter& kf,
                                int64_t frame_id) {
    const cv::Vec4f meas_xyah = tlwh_to_xyah(detection.tlwh);
    kf.update(track.mean, track.covariance, meas_xyah);
    track.tlwh = xyah_to_tlwh(track.mean);

    track.frame_id = frame_id;
    track.tracklet_len += 1;
    track.state = TrackState::Tracked;
    track.is_activated = true;
    track.score = detection.score;
}

static inline void reactivate_track(Track& track,
                                    const Track& detection,
                                    const ByteKalmanFilter& kf,
                                    int64_t frame_id) {
    update_track(track, detection, kf, frame_id);
}

static std::vector<Box> track_boxes(const std::vector<TrackPtr>& tracks) {
    std::vector<Box> out;
    out.reserve(tracks.size());
    for (const auto& t : tracks) out.push_back(t ? t->tlwh : Box{});
    return out;
}

static std::vector<TrackPtr> joint_tracks(const std::vector<TrackPtr>& a,
                                          const std::vector<TrackPtr>& b) {
    std::vector<TrackPtr> out;
    out.reserve(a.size() + b.size());
    std::unordered_set<int> seen;
    for (const auto& t : a) if (t && seen.insert(t->id).second) out.push_back(t);
    for (const auto& t : b) if (t && seen.insert(t->id).second) out.push_back(t);
    return out;
}

static std::vector<TrackPtr> sub_tracks(const std::vector<TrackPtr>& a,
                                        const std::vector<TrackPtr>& b) {
    std::unordered_set<int> drop;
    for (const auto& t : b) if (t) drop.insert(t->id);

    std::vector<TrackPtr> out;
    out.reserve(a.size());
    for (const auto& t : a) if (t && drop.find(t->id) == drop.end()) out.push_back(t);
    return out;
}

static std::pair<std::vector<TrackPtr>, std::vector<TrackPtr>> remove_duplicate_tracks(
    const std::vector<TrackPtr>& tracked,
    const std::vector<TrackPtr>& lost,
    float duplicate_iou_thresh) {

    if (tracked.empty() || lost.empty()) return {tracked, lost};

    const float duplicate_iou = std::clamp(duplicate_iou_thresh, 0.0f, 0.999f);

    std::unordered_set<int> drop_tracked;
    std::unordered_set<int> drop_lost;
    for (size_t i = 0; i < tracked.size(); ++i) {
        for (size_t j = 0; j < lost.size(); ++j) {
            if (box_iou(tracked[i]->tlwh, lost[j]->tlwh) < duplicate_iou) continue;

            const int tracked_len = tracked[i]->frame_id - tracked[i]->start_frame;
            const int lost_len = lost[j]->frame_id - lost[j]->start_frame;
            if (tracked_len > lost_len) drop_lost.insert(lost[j]->id);
            else drop_tracked.insert(tracked[i]->id);
        }
    }

    std::vector<TrackPtr> tracked_out;
    tracked_out.reserve(tracked.size());
    for (const auto& t : tracked) if (t && drop_tracked.find(t->id) == drop_tracked.end()) tracked_out.push_back(t);

    std::vector<TrackPtr> lost_out;
    lost_out.reserve(lost.size());
    for (const auto& t : lost) if (t && drop_lost.find(t->id) == drop_lost.end()) lost_out.push_back(t);

    return {std::move(tracked_out), std::move(lost_out)};
}

class ByteTracker final : public ITracker {
public:
    explicit ByteTracker(ByteTrackModuleConfig cfg)
        : cfg_(std::move(cfg)),
          scene_grid_(cfg_.scene_grid) {

        cfg_.high_thresh = std::clamp(cfg_.high_thresh, 0.01f, 0.99f);
        cfg_.low_thresh = std::clamp(cfg_.low_thresh, 0.0f, cfg_.high_thresh);

        cfg_.new_track_thresh = std::clamp(cfg_.new_track_thresh, cfg_.high_thresh, 0.999f);

        cfg_.match_iou_thresh = std::clamp(cfg_.match_iou_thresh, 0.01f, 0.99f);
        cfg_.low_match_iou_thresh = std::clamp(cfg_.low_match_iou_thresh, 0.01f, 0.99f);
        cfg_.unconfirmed_match_iou_thresh = std::clamp(cfg_.unconfirmed_match_iou_thresh, 0.01f, 0.99f);

        cfg_.track_buffer = std::max(1, cfg_.track_buffer);
        cfg_.min_box_area = std::max(0.0f, cfg_.min_box_area);
        max_time_lost_ = cfg_.track_buffer;
    }

    std::vector<Box> update(const TrackerFrameInfo& frame,
                            const std::vector<Box>& detections) override {
        frame_id_ = frame.frame_id > 0 ? frame.frame_id : frame_id_ + 1;
        scene_grid_.begin_frame(frame.stream_id);

        std::vector<TrackPtr> high_dets;
        std::vector<TrackPtr> low_dets;
        high_dets.reserve(detections.size());
        low_dets.reserve(detections.size());

        for (const auto& det : detections) {
            if (det.w <= 1.0f || det.h <= 1.0f) continue;
            if (area_of(det) < cfg_.min_box_area) continue;

            auto t = std::make_shared<Track>();
            t->tlwh = det;
            t->score = det.score;

            if (det.score >= cfg_.high_thresh) high_dets.push_back(std::move(t));
            else if (det.score >= cfg_.low_thresh) low_dets.push_back(std::move(t));
        }

        std::vector<TrackPtr> activated_tracks;
        std::vector<TrackPtr> refind_tracks;
        std::vector<TrackPtr> lost_tracks;
        std::vector<TrackPtr> removed_tracks;

        // split tracked into confirmed / unconfirmed
        std::vector<TrackPtr> unconfirmed;
        std::vector<TrackPtr> tracked_tracks;
        for (const auto& t : tracked_tracks_) {
            if (!t) continue;
            if (!t->is_activated) unconfirmed.push_back(t);
            else tracked_tracks.push_back(t);
        }

        // pool = tracked + lost, then predict
        std::vector<TrackPtr> track_pool = joint_tracks(tracked_tracks, lost_tracks_);
        for (auto& t : track_pool) if (t) predict_track(*t, kf_);

        // 1) match pool with high dets
        std::vector<std::pair<int, int>> matches;
        std::vector<int> unmatched_track_idx;
        std::vector<int> unmatched_det_idx;

        const SceneGrid* grid = (cfg_.scene_grid.enabled && frame.width > 0 && frame.height > 0) ? &scene_grid_ : nullptr;

        AssociationResult assoc = associate_detections(
            track_boxes(track_pool),
            track_boxes(high_dets),
            AssociationOptions{
                AssociationStage::High,
                cfg_.match_iou_thresh,
                cfg_.fuse_score,
                grid,
                frame.stream_id,
                frame.width,
                frame.height,
            });
        for (const auto& m : assoc.matches) matches.emplace_back(m.track_index, m.detection_index);
        unmatched_track_idx = std::move(assoc.unmatched_tracks);
        unmatched_det_idx = std::move(assoc.unmatched_detections);

        for (const auto& m : matches) {
            TrackPtr& track = track_pool[static_cast<size_t>(m.first)];
            const TrackPtr& det = high_dets[static_cast<size_t>(m.second)];
            if (!track || !det) continue;

            const Box before_update = track->tlwh;
            if (track->state == TrackState::Tracked) {
                update_track(*track, *det, kf_, frame_id_);
                activated_tracks.push_back(track);
            } else {
                reactivate_track(*track, *det, kf_, frame_id_);
                refind_tracks.push_back(track);
            }
            scene_grid_.observe_transition(frame.stream_id, frame.width, frame.height, before_update, det->tlwh);
        }

        // remaining tracked-only for low association
        std::vector<TrackPtr> remaining_tracked;
        remaining_tracked.reserve(unmatched_track_idx.size());
        for (int idx : unmatched_track_idx) {
            if (idx < 0 || idx >= static_cast<int>(track_pool.size())) continue;
            TrackPtr& t = track_pool[static_cast<size_t>(idx)];
            if (t && t->state == TrackState::Tracked) remaining_tracked.push_back(t);
        }

        // 2) match remaining tracked with low dets
        matches.clear();
        std::vector<int> unmatched_remaining_idx;
        std::vector<int> unmatched_low_det_idx;

        assoc = associate_detections(
            track_boxes(remaining_tracked),
            track_boxes(low_dets),
            AssociationOptions{
                AssociationStage::Low,
                cfg_.low_match_iou_thresh,
                cfg_.fuse_score,
                grid,
                frame.stream_id,
                frame.width,
                frame.height,
            });
        for (const auto& m : assoc.matches) matches.emplace_back(m.track_index, m.detection_index);
        unmatched_remaining_idx = std::move(assoc.unmatched_tracks);
        unmatched_low_det_idx = std::move(assoc.unmatched_detections);

        for (const auto& m : matches) {
            TrackPtr& track = remaining_tracked[static_cast<size_t>(m.first)];
            const TrackPtr& det = low_dets[static_cast<size_t>(m.second)];
            if (!track || !det) continue;
            const Box before_update = track->tlwh;
            update_track(*track, *det, kf_, frame_id_);
            activated_tracks.push_back(track);
            scene_grid_.observe_transition(frame.stream_id, frame.width, frame.height, before_update, det->tlwh);
        }

        // unmatched remaining tracked => Lost
        for (int idx : unmatched_remaining_idx) {
            if (idx < 0 || idx >= static_cast<int>(remaining_tracked.size())) continue;
            TrackPtr& track = remaining_tracked[static_cast<size_t>(idx)];
            if (!track) continue;
            track->state = TrackState::Lost;
            lost_tracks.push_back(track);
        }

        // build unmatched high det list
        std::vector<TrackPtr> unmatched_high_dets;
        unmatched_high_dets.reserve(unmatched_det_idx.size());
        for (int idx : unmatched_det_idx) {
            if (idx < 0 || idx >= static_cast<int>(high_dets.size())) continue;
            unmatched_high_dets.push_back(high_dets[static_cast<size_t>(idx)]);
        }

        // 3) match unconfirmed with remaining high dets
        matches.clear();
        std::vector<int> unmatched_unconfirmed_idx;
        std::vector<int> unmatched_unconfirmed_det_idx;

        assoc = associate_detections(
            track_boxes(unconfirmed),
            track_boxes(unmatched_high_dets),
            AssociationOptions{
                AssociationStage::Unconfirmed,
                cfg_.unconfirmed_match_iou_thresh,
                cfg_.fuse_score,
                grid,
                frame.stream_id,
                frame.width,
                frame.height,
            });
        for (const auto& m : assoc.matches) matches.emplace_back(m.track_index, m.detection_index);
        unmatched_unconfirmed_idx = std::move(assoc.unmatched_tracks);
        unmatched_unconfirmed_det_idx = std::move(assoc.unmatched_detections);

        for (const auto& m : matches) {
            TrackPtr& track = unconfirmed[static_cast<size_t>(m.first)];
            const TrackPtr& det = unmatched_high_dets[static_cast<size_t>(m.second)];
            if (!track || !det) continue;
            const Box before_update = track->tlwh;
            update_track(*track, *det, kf_, frame_id_);
            activated_tracks.push_back(track);
            scene_grid_.observe_transition(frame.stream_id, frame.width, frame.height, before_update, det->tlwh);
        }

        for (int idx : unmatched_unconfirmed_idx) {
            if (idx < 0 || idx >= static_cast<int>(unconfirmed.size())) continue;
            TrackPtr& track = unconfirmed[static_cast<size_t>(idx)];
            if (!track) continue;
            track->state = TrackState::Removed;
            removed_tracks.push_back(track);
        }

        // 4) init new tracks from remaining unmatched high dets
        for (int idx : unmatched_unconfirmed_det_idx) {
            if (idx < 0 || idx >= static_cast<int>(unmatched_high_dets.size())) continue;
            TrackPtr& det = unmatched_high_dets[static_cast<size_t>(idx)];
            if (!det) continue;
            if (det->score < cfg_.new_track_thresh) continue;
            activate_track(*det, kf_, frame_id_, next_track_id_++);
            activated_tracks.push_back(det);
            scene_grid_.observe(frame.stream_id, frame.width, frame.height, det->tlwh);
        }

        // 5) remove too-old lost tracks
        for (const auto& track : lost_tracks_) {
            if (!track) continue;
            if (frame_id_ - track->end_frame() > max_time_lost_) {
                track->state = TrackState::Removed;
                removed_tracks.push_back(track);
            }
        }

        // 6) merge lists
        std::vector<TrackPtr> tracked_next;
        tracked_next.reserve(tracked_tracks_.size());
        for (const auto& t : tracked_tracks_) {
            if (t && t->state == TrackState::Tracked) tracked_next.push_back(t);
        }
        tracked_next = joint_tracks(tracked_next, activated_tracks);
        tracked_next = joint_tracks(tracked_next, refind_tracks);

        std::vector<TrackPtr> lost_next = sub_tracks(lost_tracks_, tracked_next);
        lost_next.insert(lost_next.end(), lost_tracks.begin(), lost_tracks.end());
        lost_next = sub_tracks(lost_next, removed_tracks);

        tracked_tracks_ = std::move(tracked_next);
        lost_tracks_ = std::move(lost_next);

        // 7) deduplicate
        auto dedup = remove_duplicate_tracks(tracked_tracks_, lost_tracks_, cfg_.duplicate_iou_thresh);
        tracked_tracks_ = std::move(dedup.first);
        lost_tracks_ = std::move(dedup.second);

        // output
        std::vector<Box> output;
        output.reserve(tracked_tracks_.size());
        for (const auto& t : tracked_tracks_) {
            if (!t || !t->is_activated || t->state != TrackState::Tracked) continue;
            Box b = t->tlwh;
            b.id = t->id;
            b.score = t->score;
            b.occluded = false;
            output.push_back(std::move(b));
        }
        return output;
    }

private:
    ByteTrackModuleConfig cfg_;
    ByteKalmanFilter kf_;
    SceneGrid scene_grid_;
    int64_t frame_id_ = 0;
    int next_track_id_ = 1;
    int max_time_lost_ = 30;

    std::vector<TrackPtr> tracked_tracks_;
    std::vector<TrackPtr> lost_tracks_;
};

} // namespace

std::unique_ptr<ITracker> create_bytetrack_tracker(const ByteTrackModuleConfig& cfg) {
    return std::make_unique<ByteTracker>(cfg);
}

} // namespace veilsight
