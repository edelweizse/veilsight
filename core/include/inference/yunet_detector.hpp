#pragma once

#include <inference/detector.hpp>

#include <memory>

namespace ss {
    class YuNetDetector final : public IDetector {
    public:
        explicit YuNetDetector(YuNetModuleConfig cfg);
        ~YuNetDetector();

        YuNetDetector(YuNetDetector&&) noexcept;
        YuNetDetector& operator=(YuNetDetector&&) noexcept;

        YuNetDetector(const YuNetDetector&) = delete;
        YuNetDetector& operator=(const YuNetDetector&) = delete;

        std::vector<Box> detect(const cv::Mat& bgr) override;

    private:
        YuNetModuleConfig cfg_;
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}
