#include <tracking/tracker.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace ss {
namespace {

enum class TrackState : uint8_t {
    Tracked = 0,
    Lost = 1,
    Removed = 2,
};

struct Track {
    int id = 0;
    int start_frame = 0;
    int frame_id = 0;
    int tracklet_len = 0;
    float score = 0.0f;
    bool is_activated = false;
    TrackState state = TrackState::Tracked;
    Box tlwh{};

    cv::Mat mean;       // 8x1 (x, y, a, h, vx, vy, va, vh)
    cv::Mat covariance; // 8x8

    int end_frame() const { return frame_id; }
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

static inline float iou_of(const Box& a, const Box& b) {
    const float ax2 = a.x + a.w, ay2 = a.y + a.h;
    const float bx2 = b.x + b.w, by2 = b.y + b.h;

    const float xx1 = std::max(a.x, b.x);
    const float yy1 = std::max(a.y, b.y);
    const float xx2 = std::min(ax2, bx2);
    const float yy2 = std::min(ay2, by2);

    const float iw = std::max(0.0f, xx2 - xx1);
    const float ih = std::max(0.0f, yy2 - yy1);
    const float inter = iw * ih;
    if (inter <= 0.0f) return 0.0f;

    const float uni = area_of(a) + area_of(b) - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
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

// Clamp a detection relative to a reference (predicted) box.
// This is the face-jitter killer (partial-face boxes jump a lot).
static inline Box clamp_det_to_ref(const Box& det, const Box& ref) {
    Box adjusted = det;

    const float ref_cx = ref.x + ref.w * 0.5f;
    const float ref_cy = ref.y + ref.h * 0.5f;
    const float det_cx = det.x + det.w * 0.5f;
    const float det_cy = det.y + det.h * 0.5f;

    const float max_shift = 0.35f * std::max(ref.w, ref.h) + 4.0f;
    const float dx = std::clamp(det_cx - ref_cx, -max_shift, max_shift);
    const float dy = std::clamp(det_cy - ref_cy, -max_shift, max_shift);

    const float new_cx = ref_cx + dx;
    const float new_cy = ref_cy + dy;

    const float min_w = std::max(1.0f, ref.w * 0.70f);
    const float max_w = std::max(min_w, ref.w * 1.45f);
    const float min_h = std::max(1.0f, ref.h * 0.70f);
    const float max_h = std::max(min_h, ref.h * 1.45f);

    adjusted.w = std::clamp(det.w, min_w, max_w);
    adjusted.h = std::clamp(det.h, min_h, max_h);
    adjusted.x = new_cx - adjusted.w * 0.5f;
    adjusted.y = new_cy - adjusted.h * 0.5f;

    return adjusted;
}

static inline void damp_or_reset_velocity(cv::Mat& mean, TrackState state) {
    if (mean.empty()) return;

    if (state != TrackState::Tracked) {
        // FIX: reset ALL velocity terms for Lost/Removed, not only vh.
        for (int i = 4; i < 8; ++i) mean.at<float>(i, 0) = 0.0f; // vx, vy, va, vh
        return;
    }

    // Mild damping even when tracked helps for faces (reduces “chasing”).
    const float damp = 0.98f;
    for (int i = 4; i < 8; ++i) mean.at<float>(i, 0) *= damp;
}

static inline void predict_track(Track& track, const ByteKalmanFilter& kf) {
    if (track.mean.empty() || track.covariance.empty()) return;

    damp_or_reset_velocity(track.mean, track.state);
    kf.predict(track.mean, track.covariance);
    track.tlwh = xyah_to_tlwh(track.mean);
}

static inline void activate_track(Track& track,
                                  const ByteKalmanFilter& kf,
                                  int frame_id,
                                  int track_id) {
    const cv::Vec4f xyah = tlwh_to_xyah(track.tlwh);
    kf.initiate(xyah, track.mean, track.covariance);
    track.tlwh = xyah_to_tlwh(track.mean);

    track.id = track_id;
    track.frame_id = frame_id;
    track.start_frame = frame_id;
    track.tracklet_len = 0;
    track.state = TrackState::Tracked;

    // FIX: activate immediately (better UX for faces; reduces “missing first frame”).
    track.is_activated = true;
}

static inline void update_track(Track& track,
                                const Track& detection,
                                const ByteKalmanFilter& kf,
                                int frame_id) {
    // Clamp detection to predicted box to suppress jitter.
    const Box det_clamped = clamp_det_to_ref(detection.tlwh, track.tlwh);

    // Blend measurement toward prediction (extra stability for faces).
    const cv::Vec4f pred_xyah = tlwh_to_xyah(track.tlwh);
    cv::Vec4f meas_xyah = tlwh_to_xyah(det_clamped);
    const float meas_alpha = 0.85f; // 1.0 = raw detection, smaller = smoother
    meas_xyah[0] = meas_alpha * meas_xyah[0] + (1.0f - meas_alpha) * pred_xyah[0];
    meas_xyah[1] = meas_alpha * meas_xyah[1] + (1.0f - meas_alpha) * pred_xyah[1];
    meas_xyah[2] = meas_alpha * meas_xyah[2] + (1.0f - meas_alpha) * pred_xyah[2];
    meas_xyah[3] = meas_alpha * meas_xyah[3] + (1.0f - meas_alpha) * pred_xyah[3];

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
                                    int frame_id) {
    update_track(track, detection, kf, frame_id);
}

// ---------------------------
// Matching helpers
// ---------------------------
static inline std::vector<std::vector<float>> iou_distance(const std::vector<TrackPtr>& tracks,
                                                           const std::vector<TrackPtr>& detections) {
    if (tracks.empty() || detections.empty()) return {};
    std::vector<std::vector<float>> dists(tracks.size(), std::vector<float>(detections.size(), 1.0f));
    for (size_t i = 0; i < tracks.size(); ++i) {
        const Box t = tracks[i]->tlwh;
        for (size_t j = 0; j < detections.size(); ++j) {
            const float iou = iou_of(t, detections[j]->tlwh);
            dists[i][j] = 1.0f - iou;
        }
    }
    return dists;
}

static inline void fuse_score(std::vector<std::vector<float>>& dists,
                              const std::vector<TrackPtr>& detections) {
    for (size_t i = 0; i < dists.size(); ++i) {
        for (size_t j = 0; j < dists[i].size(); ++j) {
            const float sim = 1.0f - dists[i][j];
            const float det_score = std::clamp(detections[j]->score, 0.0f, 1.0f);
            const float fused_sim = sim * det_score;
            dists[i][j] = 1.0f - fused_sim;
        }
    }
}

// Hungarian (same as your original)
static std::vector<int> hungarian_assignment(const std::vector<std::vector<float>>& cost) {
    const int rows = static_cast<int>(cost.size());
    const int cols = rows > 0 ? static_cast<int>(cost[0].size()) : 0;
    if (rows == 0 || cols == 0) return {};

    bool transposed = false;
    std::vector<std::vector<float>> mat = cost;
    int n = rows;
    int m = cols;
    if (n > m) {
        transposed = true;
        std::vector<std::vector<float>> tmat(static_cast<size_t>(m),
                                             std::vector<float>(static_cast<size_t>(n), 0.0f));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                tmat[static_cast<size_t>(j)][static_cast<size_t>(i)] = mat[static_cast<size_t>(i)][static_cast<size_t>(j)];
        mat = std::move(tmat);
        std::swap(n, m);
    }

    const float inf = 1e9f;
    std::vector<float> u(static_cast<size_t>(n + 1), 0.0f);
    std::vector<float> v(static_cast<size_t>(m + 1), 0.0f);
    std::vector<int> p(static_cast<size_t>(m + 1), 0);
    std::vector<int> way(static_cast<size_t>(m + 1), 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(static_cast<size_t>(m + 1), inf);
        std::vector<char> used(static_cast<size_t>(m + 1), 0);

        do {
            used[static_cast<size_t>(j0)] = 1;
            const int i0 = p[static_cast<size_t>(j0)];
            float delta = inf;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[static_cast<size_t>(j)]) continue;
                const float cur =
                    mat[static_cast<size_t>(i0 - 1)][static_cast<size_t>(j - 1)] -
                    u[static_cast<size_t>(i0)] - v[static_cast<size_t>(j)];
                if (cur < minv[static_cast<size_t>(j)]) {
                    minv[static_cast<size_t>(j)] = cur;
                    way[static_cast<size_t>(j)] = j0;
                }
                if (minv[static_cast<size_t>(j)] < delta) {
                    delta = minv[static_cast<size_t>(j)];
                    j1 = j;
                }
            }

            for (int j = 0; j <= m; ++j) {
                if (used[static_cast<size_t>(j)]) {
                    u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
                    v[static_cast<size_t>(j)] -= delta;
                } else {
                    minv[static_cast<size_t>(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[static_cast<size_t>(j0)] != 0);

        do {
            const int j1 = way[static_cast<size_t>(j0)];
            p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment;
    if (!transposed) {
        assignment.assign(static_cast<size_t>(rows), -1);
        for (int j = 1; j <= m; ++j)
            if (p[static_cast<size_t>(j)] > 0)
                assignment[static_cast<size_t>(p[static_cast<size_t>(j)] - 1)] = j - 1;
        return assignment;
    }

    assignment.assign(static_cast<size_t>(rows), -1);
    for (int j = 1; j <= m; ++j) {
        const int row_t = p[static_cast<size_t>(j)] - 1;
        if (row_t < 0) continue;
        const int original_col = row_t;
        const int original_row = j - 1;
        if (original_row >= 0 && original_row < rows)
            assignment[static_cast<size_t>(original_row)] = original_col;
    }
    return assignment;
}

static void linear_assignment(const std::vector<std::vector<float>>& cost,
                              float thresh,
                              std::vector<std::pair<int, int>>& matches,
                              std::vector<int>& unmatched_rows,
                              std::vector<int>& unmatched_cols) {
    matches.clear();
    unmatched_rows.clear();
    unmatched_cols.clear();

    const int rows = static_cast<int>(cost.size());
    const int cols = rows > 0 ? static_cast<int>(cost[0].size()) : 0;
    if (rows == 0) {
        for (int j = 0; j < cols; ++j) unmatched_cols.push_back(j);
        return;
    }
    if (cols == 0) {
        for (int i = 0; i < rows; ++i) unmatched_rows.push_back(i);
        return;
    }

    const std::vector<int> assignment = hungarian_assignment(cost);
    std::vector<char> col_taken(static_cast<size_t>(cols), 0);
    for (int i = 0; i < rows; ++i) {
        if (i >= static_cast<int>(assignment.size())) {
            unmatched_rows.push_back(i);
            continue;
        }
        const int j = assignment[static_cast<size_t>(i)];
        if (j < 0 || j >= cols || cost[static_cast<size_t>(i)][static_cast<size_t>(j)] > thresh) {
            unmatched_rows.push_back(i);
            continue;
        }
        matches.emplace_back(i, j);
        col_taken[static_cast<size_t>(j)] = 1;
    }
    for (int j = 0; j < cols; ++j)
        if (!col_taken[static_cast<size_t>(j)]) unmatched_cols.push_back(j);
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

    const auto dists = iou_distance(tracked, lost);
    const float duplicate_dist_thresh = 1.0f - std::clamp(duplicate_iou_thresh, 0.0f, 0.999f);

    std::unordered_set<int> drop_tracked;
    std::unordered_set<int> drop_lost;
    for (size_t i = 0; i < dists.size(); ++i) {
        for (size_t j = 0; j < dists[i].size(); ++j) {
            if (dists[i][j] > duplicate_dist_thresh) continue;

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

// ---------------------------
// Fixed ByteTracker
// ---------------------------
class ByteTracker final : public ITracker {
public:
    explicit ByteTracker(ByteTrackModuleConfig cfg)
        : cfg_(std::move(cfg)) {

        cfg_.high_thresh = std::clamp(cfg_.high_thresh, 0.01f, 0.99f);
        cfg_.low_thresh = std::clamp(cfg_.low_thresh, 0.0f, cfg_.high_thresh);

        // Keep new_track_thresh reasonable (faces often have tighter score range).
        cfg_.new_track_thresh = std::clamp(cfg_.new_track_thresh, cfg_.high_thresh, 0.999f);

        cfg_.match_iou_thresh = std::clamp(cfg_.match_iou_thresh, 0.01f, 0.99f);
        cfg_.low_match_iou_thresh = std::clamp(cfg_.low_match_iou_thresh, 0.01f, 0.99f);
        cfg_.unconfirmed_match_iou_thresh = std::clamp(cfg_.unconfirmed_match_iou_thresh, 0.01f, 0.99f);

        cfg_.track_buffer = std::max(1, cfg_.track_buffer);
        cfg_.min_box_area = std::max(0.0f, cfg_.min_box_area);
        max_time_lost_ = cfg_.track_buffer;
    }

    std::vector<Box> update(const std::vector<Box>& detections) override {
        frame_id_ += 1;

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

        if (!track_pool.empty() && !high_dets.empty()) {
            auto dists = iou_distance(track_pool, high_dets);
            if (cfg_.fuse_score) fuse_score(dists, high_dets);
            linear_assignment(dists, 1.0f - cfg_.match_iou_thresh, matches, unmatched_track_idx, unmatched_det_idx);
        } else {
            if (track_pool.empty()) {
                unmatched_det_idx.reserve(high_dets.size());
                for (size_t i = 0; i < high_dets.size(); ++i) unmatched_det_idx.push_back(static_cast<int>(i));
            } else {
                unmatched_track_idx.reserve(track_pool.size());
                for (size_t i = 0; i < track_pool.size(); ++i) unmatched_track_idx.push_back(static_cast<int>(i));
            }
        }

        for (const auto& m : matches) {
            TrackPtr& track = track_pool[static_cast<size_t>(m.first)];
            const TrackPtr& det = high_dets[static_cast<size_t>(m.second)];
            if (!track || !det) continue;

            if (track->state == TrackState::Tracked) {
                update_track(*track, *det, kf_, frame_id_);
                activated_tracks.push_back(track);
            } else {
                reactivate_track(*track, *det, kf_, frame_id_);
                refind_tracks.push_back(track);
            }
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

        if (!remaining_tracked.empty() && !low_dets.empty()) {
            auto low_dists = iou_distance(remaining_tracked, low_dets);
            linear_assignment(low_dists, 1.0f - cfg_.low_match_iou_thresh, matches, unmatched_remaining_idx, unmatched_low_det_idx);
        } else {
            if (remaining_tracked.empty()) {
                unmatched_low_det_idx.reserve(low_dets.size());
                for (size_t i = 0; i < low_dets.size(); ++i) unmatched_low_det_idx.push_back(static_cast<int>(i));
            } else {
                unmatched_remaining_idx.reserve(remaining_tracked.size());
                for (size_t i = 0; i < remaining_tracked.size(); ++i) unmatched_remaining_idx.push_back(static_cast<int>(i));
            }
        }

        for (const auto& m : matches) {
            TrackPtr& track = remaining_tracked[static_cast<size_t>(m.first)];
            const TrackPtr& det = low_dets[static_cast<size_t>(m.second)];
            if (!track || !det) continue;
            update_track(*track, *det, kf_, frame_id_);
            activated_tracks.push_back(track);
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

        if (!unconfirmed.empty() && !unmatched_high_dets.empty()) {
            auto unconfirmed_dists = iou_distance(unconfirmed, unmatched_high_dets);
            if (cfg_.fuse_score) fuse_score(unconfirmed_dists, unmatched_high_dets);
            linear_assignment(unconfirmed_dists,
                              1.0f - cfg_.unconfirmed_match_iou_thresh,
                              matches,
                              unmatched_unconfirmed_idx,
                              unmatched_unconfirmed_det_idx);
        } else {
            if (unconfirmed.empty()) {
                unmatched_unconfirmed_det_idx.reserve(unmatched_high_dets.size());
                for (size_t i = 0; i < unmatched_high_dets.size(); ++i)
                    unmatched_unconfirmed_det_idx.push_back(static_cast<int>(i));
            } else {
                unmatched_unconfirmed_idx.reserve(unconfirmed.size());
                for (size_t i = 0; i < unconfirmed.size(); ++i)
                    unmatched_unconfirmed_idx.push_back(static_cast<int>(i));
            }
        }

        for (const auto& m : matches) {
            TrackPtr& track = unconfirmed[static_cast<size_t>(m.first)];
            const TrackPtr& det = unmatched_high_dets[static_cast<size_t>(m.second)];
            if (!track || !det) continue;
            update_track(*track, *det, kf_, frame_id_);
            activated_tracks.push_back(track);
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
    int frame_id_ = 0;
    int next_track_id_ = 1;
    int max_time_lost_ = 30;

    std::vector<TrackPtr> tracked_tracks_;
    std::vector<TrackPtr> lost_tracks_;
};

} // namespace

std::unique_ptr<ITracker> create_bytetrack_tracker(const ByteTrackModuleConfig& cfg) {
    return std::make_unique<ByteTracker>(cfg);
}

} // namespace ss
