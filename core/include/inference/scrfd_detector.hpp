#pragma once

#include <inference/detector.hpp>

#include <memory>

namespace ss {
    class SCRFDDetector final : public IDetector {
    public:
        explicit SCRFDDetector(SCRFDModuleConfig cfg);
        ~SCRFDDetector();

        SCRFDDetector(SCRFDDetector&&) noexcept;
        SCRFDDetector& operator=(SCRFDDetector&&) noexcept;

        SCRFDDetector(const SCRFDDetector&) = delete;
        SCRFDDetector& operator=(const SCRFDDetector&) = delete;

        std::vector<Box> detect(const cv::Mat& bgr) override;

    private:
        SCRFDModuleConfig cfg_;
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}
