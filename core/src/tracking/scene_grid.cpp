#include <tracking/scene_grid.hpp>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <numeric>

namespace veilsight {
    namespace {
        float clamp01(float v) {
            return std::clamp(v, 0.0f, 1.0f);
        }
    }

    SceneGrid::SceneGrid(SceneGridConfig cfg)
        : cfg_(cfg) {
        cfg_.rows = std::max(1, cfg_.rows);
        cfg_.cols = std::max(1, cfg_.cols);
        cfg_.association_weight = std::max(0.0f, cfg_.association_weight);
        cfg_.cell_distance_weight = std::max(0.0f, cfg_.cell_distance_weight);
        cfg_.occupancy_weight = std::max(0.0f, cfg_.occupancy_weight);
        cfg_.transition_weight = std::max(0.0f, cfg_.transition_weight);
        cfg_.max_extra_cost = std::max(0.0f, cfg_.max_extra_cost);
        cfg_.occupancy_decay = clamp01(cfg_.occupancy_decay);
        cfg_.transition_decay = clamp01(cfg_.transition_decay);
        cfg_.warmup_frames = std::max(0, cfg_.warmup_frames);
    }

    void SceneGrid::begin_frame(const std::string& stream_id) {
        if (!cfg_.enabled) return;

        StreamState& st = state_(stream_id);
        st.frames_seen += 1;
        for (float& value : st.occupancy) value *= cfg_.occupancy_decay;
        for (float& value : st.transitions) value *= cfg_.transition_decay;
    }

    void SceneGrid::observe(const std::string& stream_id, int width, int height, const Box& box) {
        if (!cfg_.enabled || width <= 0 || height <= 0) return;

        StreamState& st = state_(stream_id);
        const auto [row, col] = cell_for_box(width, height, box);
        st.occupancy[static_cast<size_t>(cell_index_(row, col))] += 1.0f;
    }

    void SceneGrid::observe_transition(const std::string& stream_id,
                                       int width,
                                       int height,
                                       const Box& from,
                                       const Box& to) {
        if (!cfg_.enabled || width <= 0 || height <= 0) return;

        StreamState& st = state_(stream_id);
        const auto [from_row, from_col] = cell_for_box(width, height, from);
        const auto [to_row, to_col] = cell_for_box(width, height, to);
        const int from_idx = cell_index_(from_row, from_col);
        const int to_idx = cell_index_(to_row, to_col);
        const int cells = cfg_.rows * cfg_.cols;
        st.transitions[static_cast<size_t>(from_idx * cells + to_idx)] += 1.0f;
        st.occupancy[static_cast<size_t>(to_idx)] += 1.0f;
    }

    std::pair<int, int> SceneGrid::cell_for_box(int width, int height, const Box& box) const {
        if (width <= 0 || height <= 0) return {0, 0};

        const float max_x = std::max(0.0f, static_cast<float>(width) - 1.0f);
        const float max_y = std::max(0.0f, static_cast<float>(height) - 1.0f);
        const float grid_x = std::clamp(box.x + box.w * 0.5f, 0.0f, max_x);
        const float grid_y = std::clamp(box.y + box.h * 0.95f, 0.0f, max_y);

        int col = static_cast<int>(std::floor((grid_x / static_cast<float>(width)) * cfg_.cols));
        int row = static_cast<int>(std::floor((grid_y / static_cast<float>(height)) * cfg_.rows));
        col = std::clamp(col, 0, cfg_.cols - 1);
        row = std::clamp(row, 0, cfg_.rows - 1);
        return {row, col};
    }

    SceneGridAssociationCost SceneGrid::association_cost(const std::string& stream_id,
                                                         int width,
                                                         int height,
                                                         const Box& track,
                                                         const Box& detection) const {
        SceneGridAssociationCost out;
        if (!cfg_.enabled || width <= 0 || height <= 0) return out;

        const StreamState* st = find_state_(stream_id);
        if (!st || st->frames_seen < cfg_.warmup_frames) return out;

        const auto [from_row, from_col] = cell_for_box(width, height, track);
        const auto [to_row, to_col] = cell_for_box(width, height, detection);
        const int from_idx = cell_index_(from_row, from_col);
        const int to_idx = cell_index_(to_row, to_col);
        const int cells = cfg_.rows * cfg_.cols;

        const float dr = static_cast<float>(to_row - from_row);
        const float dc = static_cast<float>(to_col - from_col);
        const float max_dist = std::max(
            1.0f,
            std::sqrt(static_cast<float>((cfg_.rows - 1) * (cfg_.rows - 1) +
                                         (cfg_.cols - 1) * (cfg_.cols - 1))));
        const float normalized_cell_distance = clamp01(std::sqrt(dr * dr + dc * dc) / max_dist);

        const float max_occupancy = st->occupancy.empty()
            ? 0.0f
            : *std::max_element(st->occupancy.begin(), st->occupancy.end());
        const float normalized_occupancy = max_occupancy > 0.0f
            ? clamp01(st->occupancy[static_cast<size_t>(to_idx)] / max_occupancy)
            : 0.0f;

        const auto row_begin = st->transitions.begin() + static_cast<std::ptrdiff_t>(from_idx * cells);
        const auto row_end = row_begin + cells;
        const float transition_sum = std::accumulate(row_begin, row_end, 0.0f);
        const float transition_probability = transition_sum > 0.0f
            ? clamp01(st->transitions[static_cast<size_t>(from_idx * cells + to_idx)] / transition_sum)
            : 0.0f;

        out.raw_cost =
            cfg_.cell_distance_weight * normalized_cell_distance +
            cfg_.occupancy_weight * (1.0f - normalized_occupancy) +
            cfg_.transition_weight * (1.0f - transition_probability);
        out.extra_cost = std::clamp(cfg_.association_weight * out.raw_cost, 0.0f, cfg_.max_extra_cost);
        return out;
    }

    int SceneGrid::cell_index_(int row, int col) const {
        return std::clamp(row, 0, cfg_.rows - 1) * cfg_.cols + std::clamp(col, 0, cfg_.cols - 1);
    }

    SceneGrid::StreamState& SceneGrid::state_(const std::string& stream_id) {
        StreamState& st = streams_[stream_id];
        const int cells = cfg_.rows * cfg_.cols;
        if (static_cast<int>(st.occupancy.size()) != cells) {
            st.occupancy.assign(static_cast<size_t>(cells), 0.0f);
        }
        if (static_cast<int>(st.transitions.size()) != cells * cells) {
            st.transitions.assign(static_cast<size_t>(cells * cells), 0.0f);
        }
        return st;
    }

    const SceneGrid::StreamState* SceneGrid::find_state_(const std::string& stream_id) const {
        const auto it = streams_.find(stream_id);
        return it == streams_.end() ? nullptr : &it->second;
    }
}
