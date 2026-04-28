#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <pipeline/metrics.hpp>
#include <pipeline/types.hpp>

namespace veilsight {
    class IStreamPublisher {
    public:
        virtual ~IStreamPublisher() = default;

        virtual void register_stream(const std::string& stream_key) = 0;
        virtual void publish_frame(const std::string& stream_key,
                                   const cv::Mat& frame,
                                   int quality) = 0;
        virtual void publish_metadata(const std::string& stream_key, std::string json) = 0;
        virtual void publish_metrics(std::string json) = 0;
    };

    class ITelemetryPublisher {
    public:
        virtual ~ITelemetryPublisher() = default;

        virtual void publish_frame_analytics(const FrameCtx& frame,
                                             const std::vector<Box>& tracks) = 0;
        virtual void publish_metrics_snapshot(const RuntimeMetrics::Snapshot& snapshot,
                                              const std::map<std::string, QueueSnapshot>& queues) = 0;
        virtual void publish_status(std::string state, std::string message) = 0;
        virtual void publish_log(std::string level, std::string message) = 0;
    };

    class NullTelemetryPublisher final : public ITelemetryPublisher {
    public:
        void publish_frame_analytics(const FrameCtx&, const std::vector<Box>&) override {}
        void publish_metrics_snapshot(const RuntimeMetrics::Snapshot&,
                                      const std::map<std::string, QueueSnapshot>&) override {}
        void publish_status(std::string, std::string) override {}
        void publish_log(std::string, std::string) override {}
    };
}
