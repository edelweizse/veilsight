#include <face_detector/yunet_detector.hpp>

#include <common/ncnn_options.hpp>

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

namespace veilsight {
    namespace {
        float area_of(const RectF& b) {
            return std::max(0.0f, b.w) * std::max(0.0f, b.h);
        }

        float iou_of(const RectF& a, const RectF& b) {
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

        float read_tensor_component(const ncnn::Mat& mat, int idx, int component, int component_count, int expected_count) {
            if (component_count <= 0) return 0.0f;
            const float* data = static_cast<const float*>(mat.data);
            if (!data || idx < 0 || component < 0 || component >= component_count) return 0.0f;

            if (mat.dims == 2) {
                if (mat.h == component_count && idx < mat.w) {
                    return data[component * mat.w + idx];
                }
                if (mat.w == component_count && idx < mat.h) {
                    return data[idx * mat.w + component];
                }
            }

            if (mat.dims == 3) {
                if (mat.c == component_count) {
                    const int plane = mat.w * mat.h;
                    if (idx < plane) {
                        const float* ch = mat.channel(component);
                        return ch[idx];
                    }
                }
                if (mat.c == 1 && mat.h == component_count && idx < mat.w) {
                    return data[component * mat.w + idx];
                }
            }

            const size_t off_by_cols = static_cast<size_t>(idx) * static_cast<size_t>(component_count) +
                                       static_cast<size_t>(component);
            if (off_by_cols < mat.total()) {
                return data[off_by_cols];
            }

            const size_t off_by_rows = static_cast<size_t>(component) * static_cast<size_t>(expected_count) +
                                       static_cast<size_t>(idx);
            if (off_by_rows < mat.total()) {
                return data[off_by_rows];
            }

            return 0.0f;
        }

        Box face_to_box(const FaceObservation& face) {
            Box box;
            box.x = face.bbox.x;
            box.y = face.bbox.y;
            box.w = face.bbox.w;
            box.h = face.bbox.h;
            box.score = face.score;
            box.face = face;
            return box;
        }
    } // namespace

    class YuNetDetector::Impl {
    public:
        explicit Impl(const YuNetModuleConfig& cfg) {
            configure_ncnn_net(net_, cfg.ncnn_threads);
            workspace_pool_allocator_.set_size_compare_ratio(0.0f);

            const std::string param = resolve_path_or_throw(cfg.param_path);
            const std::string bin = resolve_path_or_throw(cfg.bin_path);

            if (net_.load_param(param.c_str()) != 0) {
                throw std::runtime_error("Failed to load YuNet param: " + param);
            }
            if (net_.load_model(bin.c_str()) != 0) {
                throw std::runtime_error("Failed to load YuNet weights: " + bin);
            }
        }

        std::vector<FaceObservation> detect_faces(const cv::Mat& bgr, const YuNetModuleConfig& cfg) {
            if (bgr.empty()) return {};

            ncnn::Mat in = ncnn::Mat::from_pixels_resize(
                bgr.data,
                ncnn::Mat::PIXEL_BGR,
                bgr.cols,
                bgr.rows,
                cfg.input_w,
                cfg.input_h);

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

            std::array<ncnn::Mat, 12> out{};
            static constexpr const char* kOutNames[12] = {
                "out0", "out1", "out2",
                "out3", "out4", "out5",
                "out6", "out7", "out8",
                "out9", "out10", "out11"
            };

            for (size_t i = 0; i < out.size(); ++i) {
                if (ex.extract(kOutNames[i], out[i]) != 0) return {};
            }

            const float sx = static_cast<float>(bgr.cols) / static_cast<float>(cfg.input_w);
            const float sy = static_cast<float>(bgr.rows) / static_cast<float>(cfg.input_h);

            std::vector<FaceObservation> candidates;
            candidates.reserve(512);

            static constexpr int kStrides[3] = {8, 16, 32};
            for (int level = 0; level < 3; ++level) {
                const int stride = kStrides[level];
                const int cols = cfg.input_w / stride;
                const int rows = cfg.input_h / stride;
                const int num = cols * rows;

                const float* cls = static_cast<const float*>(out[static_cast<size_t>(level)].data);
                const float* obj = static_cast<const float*>(out[static_cast<size_t>(3 + level)].data);
                const float* bbox = static_cast<const float*>(out[static_cast<size_t>(6 + level)].data);
                const ncnn::Mat& landmarks = out[static_cast<size_t>(9 + level)];

                if (!cls || !obj || !bbox) continue;

                for (int idx = 0; idx < num; ++idx) {
                    const float score = std::sqrt(cls[idx] * obj[idx]);
                    if (score < cfg.score_threshold) continue;

                    const int y = idx / cols;
                    const int x = idx - y * cols;

                    const float dx = bbox[idx * 4 + 0];
                    const float dy = bbox[idx * 4 + 1];
                    const float dw = bbox[idx * 4 + 2];
                    const float dh = bbox[idx * 4 + 3];

                    const float cx = (static_cast<float>(x) + dx) * static_cast<float>(stride);
                    const float cy = (static_cast<float>(y) + dy) * static_cast<float>(stride);
                    const float w = std::exp(dw) * static_cast<float>(stride);
                    const float h = std::exp(dh) * static_cast<float>(stride);

                    const float x1 = std::max(0.0f, (cx - w * 0.5f) * sx);
                    const float y1 = std::max(0.0f, (cy - h * 0.5f) * sy);
                    const float x2 = std::min(static_cast<float>(bgr.cols), (cx + w * 0.5f) * sx);
                    const float y2 = std::min(static_cast<float>(bgr.rows), (cy + h * 0.5f) * sy);
                    if (x2 <= x1 || y2 <= y1) continue;

                    FaceObservation b;
                    b.bbox.x = x1;
                    b.bbox.y = y1;
                    b.bbox.w = x2 - x1;
                    b.bbox.h = y2 - y1;
                    b.score = score;
                    if (landmarks.data && static_cast<int>(landmarks.total()) >= num * 10) {
                        b.landmark_count = 5;
                        for (int k = 0; k < 5; ++k) {
                            const float lx = (static_cast<float>(x) +
                                              read_tensor_component(landmarks, idx, k * 2, 10, num)) *
                                             static_cast<float>(stride);
                            const float ly = (static_cast<float>(y) +
                                              read_tensor_component(landmarks, idx, k * 2 + 1, 10, num)) *
                                             static_cast<float>(stride);
                            b.landmarks[static_cast<size_t>(k)].x =
                                std::clamp(lx * sx, 0.0f, static_cast<float>(bgr.cols));
                            b.landmarks[static_cast<size_t>(k)].y =
                                std::clamp(ly * sy, 0.0f, static_cast<float>(bgr.rows));
                        }
                    }
                    candidates.push_back(std::move(b));
                }
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

            std::vector<FaceObservation> out_boxes;
            out_boxes.reserve(order.size());
            std::vector<int> keep_indices;
            keep_indices.reserve(order.size());

            for (int idx : order) {
                const FaceObservation& cand = candidates[static_cast<size_t>(idx)];
                bool keep = true;
                for (int kept : keep_indices) {
                    if (iou_of(cand.bbox, candidates[static_cast<size_t>(kept)].bbox) > cfg.nms_threshold) {
                        keep = false;
                        break;
                    }
                }
                if (keep) keep_indices.push_back(idx);
            }

            out_boxes.reserve(keep_indices.size());
            for (int idx : keep_indices) {
                out_boxes.push_back(candidates[static_cast<size_t>(idx)]);
            }
            return out_boxes;
        }

    private:
        ncnn::Net net_;
        mutable ncnn::PoolAllocator workspace_pool_allocator_;
    };

    YuNetDetector::YuNetDetector(YuNetModuleConfig cfg)
        : cfg_(std::move(cfg)),
          impl_(std::make_unique<Impl>(cfg_)) {}

    YuNetDetector::~YuNetDetector() = default;
    YuNetDetector::YuNetDetector(YuNetDetector&&) noexcept = default;
    YuNetDetector& YuNetDetector::operator=(YuNetDetector&&) noexcept = default;

    std::vector<Box> YuNetDetector::detect(const cv::Mat& bgr) {
        const auto faces = impl_->detect_faces(bgr, cfg_);
        std::vector<Box> boxes;
        boxes.reserve(faces.size());
        for (const auto& face : faces) {
            boxes.push_back(face_to_box(face));
        }
        return boxes;
    }

    std::vector<FaceObservation> YuNetDetector::detect_faces(const cv::Mat& bgr,
                                                             const FaceDetectorRunConfig& run) {
        YuNetModuleConfig run_cfg = cfg_;
        run_cfg.input_w = std::max(1, run.input_w);
        run_cfg.input_h = std::max(1, run.input_h);
        return impl_->detect_faces(bgr, run_cfg);
    }
}
