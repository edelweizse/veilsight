#include <inference/detector.hpp>

#include <inference/scrfd_detector.hpp>
#include <inference/yunet_detector.hpp>
#include <inference/yolox_detector.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace veilsight {
    namespace {
        class YuNetDetectorFactory final : public IDetectorFactory {
        public:
            explicit YuNetDetectorFactory(YuNetModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IDetector> create() const override {
                return std::make_unique<YuNetDetector>(cfg_);
            }

            int backend_threads() const override {
                return std::max(1, cfg_.ncnn_threads);
            }

        private:
            YuNetModuleConfig cfg_;
        };

        class SCRFDDetectorFactory final : public IDetectorFactory {
        public:
            explicit SCRFDDetectorFactory(SCRFDModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IDetector> create() const override {
                return std::make_unique<SCRFDDetector>(cfg_);
            }

            int backend_threads() const override {
                return std::max(1, cfg_.ncnn_threads);
            }

        private:
            SCRFDModuleConfig cfg_;
        };

        class YoloXDetectorFactory final : public IDetectorFactory {
        public:
            explicit YoloXDetectorFactory(YoloXModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IDetector> create() const override {
                return std::make_unique<YoloXDetector>(cfg_);
            }

            int backend_threads() const override {
                return std::max(1, cfg_.ncnn_threads);
            }

        private:
            YoloXModuleConfig cfg_;
        };
    }

    std::unique_ptr<IDetectorFactory> create_detector_factory(const DetectorModuleConfig& cfg) {
        if (cfg.type.empty() || cfg.type == "yolox") {
            return std::make_unique<YoloXDetectorFactory>(cfg.yolox);
        }
        if (cfg.type == "yunet") {
            return std::make_unique<YuNetDetectorFactory>(cfg.yunet);
        }
        if (cfg.type == "scrfd") {
            return std::make_unique<SCRFDDetectorFactory>(cfg.scrfd);
        }
        throw std::invalid_argument("[Detector] Unsupported detector type: " + cfg.type);
    }

    std::unique_ptr<IDetector> create_detector(const DetectorModuleConfig& cfg) {
        return create_detector_factory(cfg)->create();
    }
}
