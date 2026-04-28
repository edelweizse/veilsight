#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <common/config.hpp>
#include <pipeline/runtime.hpp>
#include <streaming/composite_stream_publisher.hpp>
#include <streaming/mjpeg_publisher.hpp>
#include <streaming/runner_http_server.hpp>
#include <streaming/webrtc_publisher.hpp>

#include "telemetry_hub.hpp"
#include "veilsight/runner/v1/runner.pb.h"

namespace veilsight {
    class RunnerManager {
    public:
        RunnerManager(std::string config_path, AppConfig config, TelemetryHub& telemetry);

        bool start(std::string* error);
        void stop();
        bool reload_config_yaml(const std::string& config_yaml, std::string* error);
        bool validate_config_yaml(const std::string& config_yaml, std::vector<std::string>* errors) const;

        runner::v1::PipelineStatus status(std::string message = {}) const;
        runner::v1::StreamList streams() const;

    private:
        static PipelineRuntime::Options make_options_(const AppConfig& config);
        static std::vector<std::string> stream_ids_(const std::vector<IngestConfig>& streams);

        bool start_locked_(std::string* error);
        void stop_locked_();
        void register_streams_locked_();

        std::string config_path_;
        AppConfig config_;
        std::vector<IngestConfig> expanded_streams_;
        std::string state_ = "stopped";
        std::string last_error_;

        mutable std::mutex mutex_;
        RunnerHTTPServer http_server_;
        MJPEGPublisher mjpeg_publisher_;
        WebRTCPublisher webrtc_publisher_;
        CompositeStreamPublisher stream_publisher_;
        TelemetryHub& telemetry_;
        std::unique_ptr<PipelineRuntime> runtime_;
    };
}
