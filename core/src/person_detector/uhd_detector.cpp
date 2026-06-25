#include <person_detector/uhd_detector.hpp>

#include <common/ncnn_options.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include <ncnn/allocator.h>
#include <ncnn/net.h>
#include <opencv2/imgproc.hpp>

namespace veilsight {
    namespace {
        static constexpr const char* kSupportedVariant = "s_anc8_w80_64x64_opencv_inter_nearest_static_nopost";
        static constexpr const char* kDefaultParamPath =
            "models/people_detectors/UHD/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_static_nopost.ncnn.param";
        static constexpr const char* kDefaultBinPath =
            "models/people_detectors/UHD/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_static_nopost.ncnn.bin";

        enum class OutputLayout {
            GatheredByAnchor,
            CatHeads,
        };

        static constexpr std::array<std::array<float, 2>, 8> kAnchors = {{
            {2.1529481273319107e-06f, 4.888256171398098e-06f},
            {4.343370164860971e-06f, 7.860499863454606e-06f},
            {5.013756890548393e-06f, 1.5962146790116094e-05f},
            {1.0511971595406067e-05f, 1.4963236935727764e-05f},
            {9.851809409155976e-06f, 3.349495818838477e-05f},
            {1.837422496464569e-05f, 5.013667396269739e-05f},
            {3.265002305852249e-05f, 7.202703272923827e-05f},
            {6.569157267222181e-05f, 8.742176578380167e-05f},
        }};

        static constexpr std::array<std::array<float, 2>, 8> kWhScale = {{
            {1.0420005321502686f, 1.092743992805481f},
            {1.16399085521698f, 1.1845228672027588f},
            {0.9121167659759521f, 1.306602120399475f},
            {1.2768088579177856f, 1.308788537979126f},
            {1.2655314207077026f, 1.4117695093154907f},
            {1.3262114524841309f, 1.4121273756027222f},
            {1.322676658630371f, 1.4722367525100708f},
            {9.637338638305664f, 10.5896577835083f},
        }};

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

        float sigmoid(float x) {
            if (x >= 0.0f) {
                const float z = std::exp(-x);
                return 1.0f / (1.0f + z);
            }
            const float z = std::exp(x);
            return z / (1.0f + z);
        }

        float softplus(float x) {
            return std::log1p(std::exp(-std::fabs(x))) + std::max(x, 0.0f);
        }

        std::string resolve_path_or_throw(const std::string& p, const char* label) {
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
            throw std::runtime_error(std::string("[UHD] ") + label + " path not found: " + p);
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

        bool has_supported_shape(const ncnn::Mat& out) {
            return out.dims == 3 && out.w == 8 && out.h == 8 && out.c == 56;
        }

        float at_chw(const ncnn::Mat& out, int channel, int y, int x) {
            const float* row = out.channel(channel).row(y);
            return row[x];
        }

        float uhd_value(const ncnn::Mat& out, OutputLayout layout, int anchor, int field, int y, int x) {
            int channel = anchor * 7 + field;
            if (layout == OutputLayout::CatHeads) {
                if (field < 4) {
                    channel = anchor * 4 + field;
                } else {
                    channel = 32 + (field - 4) * 8 + anchor;
                }
            }
            return at_chw(out, channel, y, x);
        }

        std::vector<Box> nms_sorted(std::vector<Box>& candidates, int top_k, float nms_threshold) {
            std::sort(candidates.begin(),
                      candidates.end(),
                      [](const Box& a, const Box& b) {
                          return a.score > b.score;
                      });
            if (top_k > 0 && static_cast<int>(candidates.size()) > top_k) {
                candidates.resize(static_cast<size_t>(top_k));
            }

            std::vector<Box> boxes;
            boxes.reserve(candidates.size());
            for (const auto& candidate : candidates) {
                bool keep = true;
                for (const auto& kept : boxes) {
                    if (iou_of(candidate, kept) > nms_threshold) {
                        keep = false;
                        break;
                    }
                }
                if (keep) boxes.push_back(candidate);
            }
            return boxes;
        }

        void validate_variant(const UhdModuleConfig& cfg) {
            if (cfg.variant == kSupportedVariant) return;

            const bool custom_paths = cfg.param_path != kDefaultParamPath && cfg.bin_path != kDefaultBinPath;
            if (!custom_paths) {
                throw std::runtime_error("[UHD] Unsupported variant: " + cfg.variant);
            }
        }

        struct ParamLoadPlan {
            std::string path;
            OutputLayout layout = OutputLayout::GatheredByAnchor;
            bool remove_after_load = false;
        };

        std::string patch_cat_output_param(const std::string& param_path) {
            std::ifstream in(param_path);
            if (!in.is_open()) {
                throw std::runtime_error("[UHD] Failed to open NCNN param for patching: " + param_path);
            }

            std::vector<std::string> lines;
            std::string line;
            while (std::getline(in, line)) lines.push_back(line);
            if (lines.size() < 3) {
                throw std::runtime_error("[UHD] NCNN param is too short: " + param_path);
            }

            int layer_count = 0;
            int blob_count = 0;
            {
                std::istringstream counts(lines[1]);
                counts >> layer_count >> blob_count;
                if (!counts || layer_count < 3 || blob_count < 3) {
                    throw std::runtime_error("[UHD] Failed to parse NCNN param counts: " + param_path);
                }
                lines[1] = std::to_string(layer_count - 2) + " " + std::to_string(blob_count - 2);
            }

            std::vector<std::string> patched;
            patched.reserve(lines.size() - 2);
            bool patched_concat = false;
            int removed = 0;
            for (const auto& current : lines) {
                if (current.rfind("Concat                   cat_1", 0) == 0) {
                    patched.push_back("Concat                   cat_1                    4 1 57 60 65 71 out0 0=0");
                    patched_concat = true;
                    continue;
                }
                if (current.rfind("Crop                     select_0", 0) == 0 ||
                    current.rfind("Squeeze                  squeeze_73", 0) == 0) {
                    ++removed;
                    continue;
                }
                patched.push_back(current);
            }

            if (!patched_concat || removed != 2) {
                throw std::runtime_error("[UHD] Failed to patch known NCNN no-postprocess graph tail: " + param_path);
            }

            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const std::filesystem::path out_path =
                std::filesystem::temp_directory_path() /
                ("veilsight_uhd_" + std::to_string(stamp) + ".ncnn.param");
            std::ofstream out(out_path);
            if (!out.is_open()) {
                throw std::runtime_error("[UHD] Failed to create patched NCNN param: " + out_path.string());
            }
            for (const auto& patched_line : patched) {
                out << patched_line << '\n';
            }
            return out_path.string();
        }

        ParamLoadPlan make_param_load_plan(const std::string& param_path) {
            std::ifstream in(param_path);
            if (!in.is_open()) {
                throw std::runtime_error("[UHD] Failed to open NCNN param: " + param_path);
            }
            std::ostringstream content;
            content << in.rdbuf();
            const std::string text = content.str();

            const bool has_bad_select_tail =
                text.find("Concat                   cat_1") != std::string::npos &&
                text.find("Crop                     select_0") != std::string::npos &&
                text.find("Squeeze                  squeeze_73") != std::string::npos;
            if (!has_bad_select_tail) {
                return ParamLoadPlan{param_path, OutputLayout::GatheredByAnchor, false};
            }

            return ParamLoadPlan{patch_cat_output_param(param_path), OutputLayout::CatHeads, true};
        }
    }

    class UhdDetector::Impl {
    public:
        explicit Impl(const UhdModuleConfig& cfg) {
            validate_variant(cfg);

            configure_ncnn_net(net_, cfg.ncnn_threads);
            workspace_pool_allocator_.set_size_compare_ratio(0.0f);

            const std::string param = resolve_path_or_throw(cfg.param_path, "param");
            const std::string bin = resolve_path_or_throw(cfg.bin_path, "weights");
            const ParamLoadPlan param_plan = make_param_load_plan(param);
            output_layout_ = param_plan.layout;
            if (net_.load_param(param_plan.path.c_str()) != 0) {
                if (param_plan.remove_after_load) {
                    std::filesystem::remove(param_plan.path);
                }
                throw std::runtime_error("[UHD] Failed to load NCNN param: " + param);
            }
            if (param_plan.remove_after_load) {
                std::filesystem::remove(param_plan.path);
            }
            if (net_.load_model(bin.c_str()) != 0) {
                throw std::runtime_error("[UHD] Failed to load NCNN weights: " + bin);
            }
        }

        std::vector<Box> detect(const cv::Mat& bgr, const UhdModuleConfig& cfg) {
            if (bgr.empty()) return {};

            cv::Mat input;
            cv::resize(bgr, input, cv::Size(cfg.input_w, cfg.input_h), 0.0, 0.0, cv::INTER_NEAREST);
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
            if (!has_supported_shape(out)) return {};

            std::vector<Box> candidates;
            candidates.reserve(8 * 8 * 8);

            for (int a = 0; a < 8; ++a) {
                for (int gy = 0; gy < 8; ++gy) {
                    for (int gx = 0; gx < 8; ++gx) {
                        const float tx = uhd_value(out, output_layout_, a, 0, gy, gx);
                        const float ty = uhd_value(out, output_layout_, a, 1, gy, gx);
                        const float tw = uhd_value(out, output_layout_, a, 2, gy, gx);
                        const float th = uhd_value(out, output_layout_, a, 3, gy, gx);
                        const float obj = uhd_value(out, output_layout_, a, 4, gy, gx);
                        const float quality = uhd_value(out, output_layout_, a, 5, gy, gx);
                        const float cls = uhd_value(out, output_layout_, a, 6, gy, gx);

                        const float score = sigmoid(obj) * sigmoid(quality) * sigmoid(cls);
                        if (score < cfg.score_threshold) continue;

                        const float cx_norm = (sigmoid(tx) + static_cast<float>(gx)) / 8.0f;
                        const float cy_norm = (sigmoid(ty) + static_cast<float>(gy)) / 8.0f;
                        const float w_norm = kAnchors[static_cast<size_t>(a)][0] *
                                             softplus(tw) *
                                             kWhScale[static_cast<size_t>(a)][0];
                        const float h_norm = kAnchors[static_cast<size_t>(a)][1] *
                                             softplus(th) *
                                             kWhScale[static_cast<size_t>(a)][1];

                        const float cx = cx_norm * static_cast<float>(bgr.cols);
                        const float cy = cy_norm * static_cast<float>(bgr.rows);
                        const float w = w_norm * static_cast<float>(bgr.cols);
                        const float h = h_norm * static_cast<float>(bgr.rows);

                        Box box = clip_box(
                            cx - 0.5f * w,
                            cy - 0.5f * h,
                            cx + 0.5f * w,
                            cy + 0.5f * h,
                            bgr.cols,
                            bgr.rows);
                        if (box.w <= 0.0f || box.h <= 0.0f) continue;
                        box.score = score;
                        candidates.push_back(box);
                    }
                }
            }

            return nms_sorted(candidates, cfg.top_k, cfg.nms_threshold);
        }

    private:
        ncnn::Net net_;
        OutputLayout output_layout_ = OutputLayout::GatheredByAnchor;
        mutable ncnn::PoolAllocator workspace_pool_allocator_;
    };

    UhdDetector::UhdDetector(UhdModuleConfig cfg)
        : cfg_(std::move(cfg)),
          impl_(std::make_unique<Impl>(cfg_)) {}

    UhdDetector::~UhdDetector() = default;
    UhdDetector::UhdDetector(UhdDetector&&) noexcept = default;
    UhdDetector& UhdDetector::operator=(UhdDetector&&) noexcept = default;

    std::vector<Box> UhdDetector::detect(const cv::Mat& bgr) {
        return impl_->detect(bgr, cfg_);
    }
}
