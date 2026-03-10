#pragma once

#include <common/config.hpp>
#include <pipeline/types.hpp>

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

namespace ss {
    class IDetector {
    public:
        virtual ~IDetector() = default;
        virtual std::vector<Box> detect(const cv::Mat& bgr) = 0;
    };

    class IDetectorFactory {
    public:
        virtual ~IDetectorFactory() = default;
        virtual std::unique_ptr<IDetector> create() const = 0;
        virtual int backend_threads() const = 0;
    };

    std::unique_ptr<IDetectorFactory> create_detector_factory(const DetectorModuleConfig& cfg);
    std::unique_ptr<IDetector> create_detector(const DetectorModuleConfig& cfg);
}
