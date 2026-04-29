#include <inference/yolox_detector.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ncnn/allocator.h>
#include <ncnn/layer.h>
#include <ncnn/net.h>
#include <opencv2/imgproc.hpp>

namespace veilsight {
    namespace {
        class YoloV5Focus final : public ncnn::Layer {
        public:
            YoloV5Focus() {
                one_blob_only = true;
                support_inplace = false;
            }

            int forward(const ncnn::Mat& bottom_blob,
                        ncnn::Mat& top_blob,
                        const ncnn::Option& opt) const override {
                if (bottom_blob.dims != 3) return -1;

                const int out_w = bottom_blob.w / 2;
                const int out_h = bottom_blob.h / 2;
                const int out_c = bottom_blob.c * 4;
                top_blob.create(out_w, out_h, out_c, bottom_blob.elemsize, opt.blob_allocator);
                if (top_blob.empty()) return -100;

                for (int q = 0; q < bottom_blob.c; ++q) {
                    const ncnn::Mat in = bottom_blob.channel(q);
                    float* out0 = top_blob.channel(q * 4 + 0);
                    float* out1 = top_blob.channel(q * 4 + 1);
                    float* out2 = top_blob.channel(q * 4 + 2);
                    float* out3 = top_blob.channel(q * 4 + 3);

                    for (int y = 0; y < out_h; ++y) {
                        const float* row0 = in.row(y * 2);
                        const float* row1 = in.row(y * 2 + 1);
                        for (int x = 0; x < out_w; ++x) {
                            out0[y * out_w + x] = row0[x * 2];
                            out1[y * out_w + x] = row0[x * 2 + 1];
                            out2[y * out_w + x] = row1[x * 2];
                            out3[y * out_w + x] = row1[x * 2 + 1];
                        }
                    }
                }
                return 0;
            }
        };

        DEFINE_LAYER_CREATOR(YoloV5Focus)

        struct LetterboxInfo {
            float scale_x = 1.0f;
            float scale_y = 1.0f;
            float pad_x = 0.0f;
            float pad_y = 0.0f;
        };

        struct DetectionRow {
            float values[6]{};
        };

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
            const fs::path candidates[] = {
                fs::path("../") / p,
                fs::path("../../") / p,
                fs::path("../../../") / p,
            };
            for (const auto& candidate : candidates) {
                if (fs::exists(candidate)) return candidate.string();
            }
            throw std::runtime_error("Model path not found: " + p);
        }

        cv::Mat make_input(const cv::Mat& bgr, const YoloXModuleConfig& cfg, LetterboxInfo& info) {
            if (!cfg.letterbox) {
                info.scale_x = static_cast<float>(cfg.input_w) / static_cast<float>(std::max(1, bgr.cols));
                info.scale_y = static_cast<float>(cfg.input_h) / static_cast<float>(std::max(1, bgr.rows));
                info.pad_x = 0.0f;
                info.pad_y = 0.0f;
                cv::Mat resized;
                cv::resize(bgr, resized, cv::Size(cfg.input_w, cfg.input_h));
                return resized;
            }

            const float sx = static_cast<float>(cfg.input_w) / static_cast<float>(std::max(1, bgr.cols));
            const float sy = static_cast<float>(cfg.input_h) / static_cast<float>(std::max(1, bgr.rows));
            const float scale = std::min(sx, sy);
            info.scale_x = scale;
            info.scale_y = scale;

            const int resized_w = std::max(1, static_cast<int>(std::round(static_cast<float>(bgr.cols) * scale)));
            const int resized_h = std::max(1, static_cast<int>(std::round(static_cast<float>(bgr.rows) * scale)));
            const int pad_x = std::max(0, (cfg.input_w - resized_w) / 2);
            const int pad_y = std::max(0, (cfg.input_h - resized_h) / 2);
            info.pad_x = static_cast<float>(pad_x);
            info.pad_y = static_cast<float>(pad_y);

            cv::Mat resized;
            cv::resize(bgr, resized, cv::Size(resized_w, resized_h));
            cv::Mat canvas(cfg.input_h, cfg.input_w, bgr.type(), cv::Scalar(114, 114, 114));
            resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_w, resized_h)));
            return canvas;
        }

        Box clip_box(float x1, float y1, float x2, float y2, int width, int height) {
            x1 = std::clamp(x1, 0.0f, static_cast<float>(width));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(height));
            x2 = std::clamp(x2, 0.0f, static_cast<float>(width));
            y2 = std::clamp(y2, 0.0f, static_cast<float>(height));
            if (x2 <= x1 || y2 <= y1) return {};

            Box out;
            out.x = x1;
            out.y = y1;
            out.w = x2 - x1;
            out.h = y2 - y1;
            return out;
        }

        std::vector<DetectionRow> rows_from_output(const ncnn::Mat& out) {
            std::vector<DetectionRow> rows;
            const float* data = static_cast<const float*>(out.data);
            if (!data || out.total() < 6) return rows;

            if (out.dims == 2 && out.w == 6) {
                rows.reserve(static_cast<size_t>(out.h));
                for (int y = 0; y < out.h; ++y) {
                    DetectionRow row;
                    for (int k = 0; k < 6; ++k) row.values[k] = data[static_cast<size_t>(y * out.w + k)];
                    rows.push_back(row);
                }
                return rows;
            }

            if (out.dims == 2 && out.h == 6) {
                rows.reserve(static_cast<size_t>(out.w));
                for (int x = 0; x < out.w; ++x) {
                    DetectionRow row;
                    for (int k = 0; k < 6; ++k) row.values[k] = data[static_cast<size_t>(k * out.w + x)];
                    rows.push_back(row);
                }
                return rows;
            }

            if (out.dims == 3 && out.w == 6) {
                rows.reserve(static_cast<size_t>(out.c * out.h));
                for (int q = 0; q < out.c; ++q) {
                    const float* channel = out.channel(q);
                    for (int y = 0; y < out.h; ++y) {
                        DetectionRow row;
                        for (int k = 0; k < 6; ++k) row.values[k] = channel[static_cast<size_t>(y * out.w + k)];
                        rows.push_back(row);
                    }
                }
                return rows;
            }

            const size_t count = out.total() / 6;
            rows.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                DetectionRow row;
                for (int k = 0; k < 6; ++k) row.values[k] = data[i * 6 + static_cast<size_t>(k)];
                rows.push_back(row);
            }
            return rows;
        }

        void decode_raw_yolox(float raw_x,
                              float raw_y,
                              float raw_w,
                              float raw_h,
                              size_t index,
                              int input_w,
                              int input_h,
                              int& x1,
                              int& y1,
                              float& cx,
                              float& cy,
                              float& w,
                              float& h) {
            static constexpr int kStrides[3] = {8, 16, 32};
            size_t offset = 0;
            for (int stride : kStrides) {
                const int grid_w = std::max(1, input_w / stride);
                const int grid_h = std::max(1, input_h / stride);
                const size_t level_count = static_cast<size_t>(grid_w) * static_cast<size_t>(grid_h);
                if (index < offset + level_count) {
                    const size_t local = index - offset;
                    x1 = static_cast<int>(local % static_cast<size_t>(grid_w));
                    y1 = static_cast<int>(local / static_cast<size_t>(grid_w));
                    cx = (raw_x + static_cast<float>(x1)) * static_cast<float>(stride);
                    cy = (raw_y + static_cast<float>(y1)) * static_cast<float>(stride);
                    w = std::exp(raw_w) * static_cast<float>(stride);
                    h = std::exp(raw_h) * static_cast<float>(stride);
                    return;
                }
                offset += level_count;
            }

            x1 = 0;
            y1 = 0;
            cx = raw_x;
            cy = raw_y;
            w = raw_w;
            h = raw_h;
        }
    }

    class YoloXDetector::Impl {
    public:
        explicit Impl(const YoloXModuleConfig& cfg) {
            net_.opt.use_vulkan_compute = false;
            net_.opt.num_threads = std::max(1, cfg.ncnn_threads);
            workspace_pool_allocator_.set_size_compare_ratio(0.0f);

            const std::string param = resolve_path_or_throw(cfg.param_path);
            const std::string bin = resolve_path_or_throw(cfg.bin_path);

            net_.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
            if (net_.load_param(param.c_str()) != 0) {
                throw std::runtime_error("Failed to load YOLOX NCNN param: " + param);
            }
            if (net_.load_model(bin.c_str()) != 0) {
                throw std::runtime_error("Failed to load YOLOX NCNN weights: " + bin);
            }
        }

        std::vector<Box> detect(const cv::Mat& bgr, const YoloXModuleConfig& cfg) {
            if (bgr.empty()) return {};

            LetterboxInfo letterbox;
            const cv::Mat input = make_input(bgr, cfg, letterbox);
            ncnn::Mat in = ncnn::Mat::from_pixels(
                input.data,
                ncnn::Mat::PIXEL_BGR2RGB,
                input.cols,
                input.rows);
            static const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
            in.substract_mean_normalize(nullptr, norm_vals);

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

            if (ex.input("in0", in) != 0) return {};

            ncnn::Mat out;
            if (ex.extract("out0", out) != 0) return {};

            const std::vector<DetectionRow> rows = rows_from_output(out);
            std::vector<Box> candidates;
            candidates.reserve(rows.size());

            for (size_t i = 0; i < rows.size(); ++i) {
                const float* r = rows[i].values;
                const float objectness = std::clamp(r[4], 0.0f, 1.0f);
                const float class_score = std::clamp(r[5], 0.0f, 1.0f);
                const float score = objectness * class_score;
                if (score < cfg.score_threshold) continue;

                float cx = 0.0f;
                float cy = 0.0f;
                float w = 0.0f;
                float h = 0.0f;
                if (cfg.decoded_output) {
                    cx = r[0];
                    cy = r[1];
                    w = r[2];
                    h = r[3];
                } else {
                    int grid_x = 0;
                    int grid_y = 0;
                    decode_raw_yolox(r[0],
                                     r[1],
                                     r[2],
                                     r[3],
                                     i,
                                     cfg.input_w,
                                     cfg.input_h,
                                     grid_x,
                                     grid_y,
                                     cx,
                                     cy,
                                     w,
                                     h);
                }

                float x1 = cx - w * 0.5f;
                float y1 = cy - h * 0.5f;
                float x2 = cx + w * 0.5f;
                float y2 = cy + h * 0.5f;

                x1 = (x1 - letterbox.pad_x) / letterbox.scale_x;
                x2 = (x2 - letterbox.pad_x) / letterbox.scale_x;
                y1 = (y1 - letterbox.pad_y) / letterbox.scale_y;
                y2 = (y2 - letterbox.pad_y) / letterbox.scale_y;

                Box box = clip_box(x1, y1, x2, y2, bgr.cols, bgr.rows);
                if (box.w <= 0.0f || box.h <= 0.0f) continue;
                box.score = score;
                candidates.push_back(std::move(box));
            }

            std::vector<int> order(candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i) order[i] = static_cast<int>(i);
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

            std::vector<Box> boxes;
            boxes.reserve(keep_indices.size());
            for (const int idx : keep_indices) {
                boxes.push_back(candidates[static_cast<size_t>(idx)]);
            }
            return boxes;
        }

    private:
        ncnn::Net net_;
        mutable ncnn::PoolAllocator workspace_pool_allocator_;
    };

    YoloXDetector::YoloXDetector(YoloXModuleConfig cfg)
        : cfg_(std::move(cfg)),
          impl_(std::make_unique<Impl>(cfg_)) {}

    YoloXDetector::~YoloXDetector() = default;
    YoloXDetector::YoloXDetector(YoloXDetector&&) noexcept = default;
    YoloXDetector& YoloXDetector::operator=(YoloXDetector&&) noexcept = default;

    std::vector<Box> YoloXDetector::detect(const cv::Mat& bgr) {
        return impl_->detect(bgr, cfg_);
    }
}
