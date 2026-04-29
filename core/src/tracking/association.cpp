#include <tracking/association.hpp>

#include <algorithm>
#include <utility>

namespace veilsight {
    namespace {
        float area_of(const Box& b) {
            return std::max(0.0f, b.w) * std::max(0.0f, b.h);
        }
    }

    float box_iou(const Box& a, const Box& b) {
        const float ax2 = a.x + a.w;
        const float ay2 = a.y + a.h;
        const float bx2 = b.x + b.w;
        const float by2 = b.y + b.h;

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

    std::vector<int> hungarian_assignment(const std::vector<std::vector<float>>& cost) {
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
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    tmat[static_cast<size_t>(j)][static_cast<size_t>(i)] =
                        mat[static_cast<size_t>(i)][static_cast<size_t>(j)];
                }
            }
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

        std::vector<int> assignment(static_cast<size_t>(rows), -1);
        if (!transposed) {
            for (int j = 1; j <= m; ++j) {
                if (p[static_cast<size_t>(j)] > 0) {
                    assignment[static_cast<size_t>(p[static_cast<size_t>(j)] - 1)] = j - 1;
                }
            }
            return assignment;
        }

        for (int j = 1; j <= m; ++j) {
            const int row_t = p[static_cast<size_t>(j)] - 1;
            if (row_t < 0) continue;
            const int original_col = row_t;
            const int original_row = j - 1;
            if (original_row >= 0 && original_row < rows) {
                assignment[static_cast<size_t>(original_row)] = original_col;
            }
        }
        return assignment;
    }

    AssociationResult associate_detections(const std::vector<Box>& tracks,
                                           const std::vector<Box>& detections,
                                           const AssociationOptions& options) {
        AssociationResult result;
        const int rows = static_cast<int>(tracks.size());
        const int cols = static_cast<int>(detections.size());

        if (rows == 0) {
            result.unmatched_detections.reserve(detections.size());
            for (int j = 0; j < cols; ++j) result.unmatched_detections.push_back(j);
            return result;
        }
        if (cols == 0) {
            result.unmatched_tracks.reserve(tracks.size());
            for (int i = 0; i < rows; ++i) result.unmatched_tracks.push_back(i);
            return result;
        }

        std::vector<std::vector<float>> costs(static_cast<size_t>(rows),
                                              std::vector<float>(static_cast<size_t>(cols), 1.0f));
        std::vector<std::vector<float>> ious(static_cast<size_t>(rows),
                                             std::vector<float>(static_cast<size_t>(cols), 0.0f));
        std::vector<std::vector<float>> grid_costs(static_cast<size_t>(rows),
                                                   std::vector<float>(static_cast<size_t>(cols), 0.0f));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const float iou = box_iou(tracks[static_cast<size_t>(i)], detections[static_cast<size_t>(j)]);
                const float det_score = std::clamp(detections[static_cast<size_t>(j)].score, 0.0f, 1.0f);
                float total_cost = options.fuse_score ? 1.0f - (iou * det_score) : 1.0f - iou;

                float grid_raw_cost = 0.0f;
                if (options.scene_grid) {
                    const SceneGridAssociationCost grid =
                        options.scene_grid->association_cost(options.stream_id,
                                                             options.frame_width,
                                                             options.frame_height,
                                                             tracks[static_cast<size_t>(i)],
                                                             detections[static_cast<size_t>(j)]);
                    grid_raw_cost = grid.raw_cost;
                    total_cost += grid.extra_cost;
                }

                costs[static_cast<size_t>(i)][static_cast<size_t>(j)] = total_cost;
                ious[static_cast<size_t>(i)][static_cast<size_t>(j)] = iou;
                grid_costs[static_cast<size_t>(i)][static_cast<size_t>(j)] = grid_raw_cost;
            }
        }

        const std::vector<int> assignment = hungarian_assignment(costs);
        std::vector<char> detection_taken(static_cast<size_t>(cols), 0);
        const float threshold_iou = std::clamp(options.iou_threshold, 0.0f, 0.999f);

        for (int i = 0; i < rows; ++i) {
            const int j = i < static_cast<int>(assignment.size()) ? assignment[static_cast<size_t>(i)] : -1;
            if (j < 0 || j >= cols ||
                ious[static_cast<size_t>(i)][static_cast<size_t>(j)] < threshold_iou) {
                result.unmatched_tracks.push_back(i);
                continue;
            }
            detection_taken[static_cast<size_t>(j)] = 1;
            result.matches.push_back(AssociationMatch{
                i,
                j,
                costs[static_cast<size_t>(i)][static_cast<size_t>(j)],
                ious[static_cast<size_t>(i)][static_cast<size_t>(j)],
                grid_costs[static_cast<size_t>(i)][static_cast<size_t>(j)],
                options.stage,
            });
        }

        for (int j = 0; j < cols; ++j) {
            if (!detection_taken[static_cast<size_t>(j)]) result.unmatched_detections.push_back(j);
        }
        return result;
    }
}
