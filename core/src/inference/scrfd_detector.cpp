#include <inference/scrfd_detector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ncnn/allocator.h>
#include <ncnn/net.h>

namespace ss {
    namespace {
        float area_of(const Box& b) {
            return std::max(0.0f, b.w) * std::max(0.0f, b.h);
        }

        float iou_of(const Box& a, const Box& b) {
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

        std::string resolve_path_or_throw(const std::string& p) {
            namespace fs = std::filesystem;
            if (fs::exists(fs::path(p))) return p;
            const fs::path alt = fs::path("../../../") / p;
            if (fs::exists(alt)) return alt.string();
            throw std::runtime_error("Model path not found: " + p);
        }

        int mat_total_len(const ncnn::Mat& mat) {
            return static_cast<int>(mat.total());
        }

        float read_score(const ncnn::Mat& score, int idx) {
            const float* data = static_cast<const float*>(score.data);
            if (!data || idx < 0) return 0.0f;
            const int n = mat_total_len(score);
            if (idx >= n) return 0.0f;
            return data[idx];
        }

        float read_box_component(const ncnn::Mat& box, int idx, int component, int expected_count) {
            const float* data = static_cast<const float*>(box.data);
            if (!data || idx < 0 || component < 0 || component > 3) return 0.0f;

            if (box.dims == 2) {
                if (box.h == 4 && idx < box.w) {
                    return data[component * box.w + idx];
                }
                if (box.w == 4 && idx < box.h) {
                    return data[idx * box.w + component];
                }
            }

            if (box.dims == 3) {
                if (box.c == 4) {
                    const int plane = box.w * box.h;
                    if (idx < plane) {
                        const float* ch = box.channel(component);
                        return ch[idx];
                    }
                }
                if (box.c == 1 && box.h == 4 && idx < box.w) {
                    return data[component * box.w + idx];
                }
            }

            const size_t off_by_rows = static_cast<size_t>(component) * static_cast<size_t>(expected_count) +
                                       static_cast<size_t>(idx);
            if (off_by_rows < box.total()) {
                return data[off_by_rows];
            }

            const size_t off_by_cols = static_cast<size_t>(idx) * 4u + static_cast<size_t>(component);
            if (off_by_cols < box.total()) {
                return data[off_by_cols];
            }

            return 0.0f;
        }
    } // namespace

    class SCRFDDetector::Impl {
    public:
        explicit Impl(const SCRFDModuleConfig& cfg) {
            const bool variant_supported =
                cfg.variant == "25g" ||
                cfg.variant == "500m" ||
                cfg.variant == "25g_landmarks" ||
                cfg.variant == "500m_landmarks" ||
                    cfg.variant == "10g";
            if (!variant_supported) {
                throw std::invalid_argument("[SCRFD] Unsupported variant: " + cfg.variant);
            }

            net_.opt.use_vulkan_compute = false;
            net_.opt.num_threads = std::max(1, cfg.ncnn_threads);
            workspace_pool_allocator_.set_size_compare_ratio(0.0f);

            const std::string param = resolve_path_or_throw(cfg.param_path);
            const std::string bin = resolve_path_or_throw(cfg.bin_path);

            if (net_.load_param(param.c_str()) != 0) {
                throw std::runtime_error("Failed to load SCRFD param: " + param);
            }
            if (net_.load_model(bin.c_str()) != 0) {
                throw std::runtime_error("Failed to load SCRFD weights: " + bin);
            }
        }

        std::vector<Box> detect(const cv::Mat& bgr, const SCRFDModuleConfig& cfg) {
            if (bgr.empty()) return {};

            ncnn::Mat in = ncnn::Mat::from_pixels_resize(
                bgr.data,
                ncnn::Mat::PIXEL_BGR,
                bgr.cols,
                bgr.rows,
                cfg.input_w,
                cfg.input_h);

            static const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
            static const float norm_vals[3] = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
            in.substract_mean_normalize(mean_vals, norm_vals);

            ncnn::Extractor ex = net_.create_extractor();
            ex.set_light_mode(true);

            thread_local ncnn::UnlockedPoolAllocator blob_pool_allocator;
            thread_local bool blob_pool_initialized = false;
            if (!blob_pool_initialized) {
                blob_pool_allocator.set_size_compare_ratio(0.0f);
                blob_pool_initialized = true;
            }
            ex.set_blob_allocator(&blob_pool_allocator);
            ex.set_workspace_allocator(&workspace_pool_allocator_);

            if (ex.input("in0", in) != 0) {
                return {};
            }

            std::array<ncnn::Mat, 9> out{};
            for (int i = 0; i < 6; ++i) {
                const std::string name = "out" + std::to_string(i);
                if (ex.extract(name.c_str(), out[static_cast<size_t>(i)]) != 0) {
                    return {};
                }
            }

            const bool expects_landmarks = cfg.variant.find("landmarks") != std::string::npos;
            if (expects_landmarks) {
                for (int i = 6; i < 9; ++i) {
                    const std::string name = "out" + std::to_string(i);
                    (void)ex.extract(name.c_str(), out[static_cast<size_t>(i)]);
                }
            }

            const float sx = static_cast<float>(bgr.cols) / static_cast<float>(cfg.input_w);
            const float sy = static_cast<float>(bgr.rows) / static_cast<float>(cfg.input_h);

            std::vector<Box> candidates;
            candidates.reserve(1024);

            static constexpr int kStrides[3] = {8, 16, 32};
            for (int level = 0; level < 3; ++level) {
                const int stride = kStrides[level];
                const ncnn::Mat& score = out[static_cast<size_t>(level)];
                const ncnn::Mat& box = out[static_cast<size_t>(3 + level)];

                const int feat_w = cfg.input_w / stride;
                const int feat_h = cfg.input_h / stride;
                const int locations = feat_w * feat_h;
                if (locations <= 0) continue;

                const int score_count = mat_total_len(score);
                const int box_total = mat_total_len(box);
                if (score_count <= 0 || box_total <= 0) continue;

                const int anchors = std::max(1, score_count / locations);
                const int box_count = std::max(1, box_total / 4);
                const int count = std::min(score_count, box_count);

                for (int idx = 0; idx < count; ++idx) {
                    const float score_val = read_score(score, idx);
                    if (score_val < cfg.score_threshold) continue;

                    const int pos = idx / anchors;
                    if (pos < 0 || pos >= locations) continue;

                    const int y = pos / feat_w;
                    const int x = pos - y * feat_w;
                    const float cx = (static_cast<float>(x) + 0.5f) * static_cast<float>(stride);
                    const float cy = (static_cast<float>(y) + 0.5f) * static_cast<float>(stride);

                    const float l = read_box_component(box, idx, 0, count) * static_cast<float>(stride);
                    const float t = read_box_component(box, idx, 1, count) * static_cast<float>(stride);
                    const float r = read_box_component(box, idx, 2, count) * static_cast<float>(stride);
                    const float b = read_box_component(box, idx, 3, count) * static_cast<float>(stride);

                    const float x1 = std::clamp((cx - l) * sx, 0.0f, static_cast<float>(bgr.cols));
                    const float y1 = std::clamp((cy - t) * sy, 0.0f, static_cast<float>(bgr.rows));
                    const float x2 = std::clamp((cx + r) * sx, 0.0f, static_cast<float>(bgr.cols));
                    const float y2 = std::clamp((cy + b) * sy, 0.0f, static_cast<float>(bgr.rows));
                    if (x2 <= x1 || y2 <= y1) continue;

                    Box det;
                    det.x = x1;
                    det.y = y1;
                    det.w = x2 - x1;
                    det.h = y2 - y1;
                    det.score = score_val;
                    candidates.push_back(std::move(det));
                }
            }

            std::vector<int> order(candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i) {
                order[i] = static_cast<int>(i);
            }

            std::sort(order.begin(),
                      order.end(),
                      [&candidates](int a, int b) {
                          return candidates[static_cast<size_t>(a)].score >
                                 candidates[static_cast<size_t>(b)].score;
                      });

            if (cfg.top_k > 0 && static_cast<int>(order.size()) > cfg.top_k) {
                order.resize(static_cast<size_t>(cfg.top_k));
            }

            std::vector<int> keep_indices;
            keep_indices.reserve(order.size());
            for (const int idx : order) {
                const Box& candidate = candidates[static_cast<size_t>(idx)];
                bool keep = true;
                for (const int kept : keep_indices) {
                    if (iou_of(candidate, candidates[static_cast<size_t>(kept)]) > cfg.nms_threshold) {
                        keep = false;
                        break;
                    }
                }
                if (keep) keep_indices.push_back(idx);
            }

            std::vector<Box> out_boxes;
            out_boxes.reserve(keep_indices.size());
            for (const int idx : keep_indices) {
                out_boxes.push_back(candidates[static_cast<size_t>(idx)]);
            }
            return out_boxes;
        }

    private:
        ncnn::Net net_;
        mutable ncnn::PoolAllocator workspace_pool_allocator_;
    };

    SCRFDDetector::SCRFDDetector(SCRFDModuleConfig cfg)
        : cfg_(std::move(cfg)),
          impl_(std::make_unique<Impl>(cfg_)) {}

    SCRFDDetector::~SCRFDDetector() = default;
    SCRFDDetector::SCRFDDetector(SCRFDDetector&&) noexcept = default;
    SCRFDDetector& SCRFDDetector::operator=(SCRFDDetector&&) noexcept = default;

    std::vector<Box> SCRFDDetector::detect(const cv::Mat& bgr) {
        return impl_->detect(bgr, cfg_);
    }
}
