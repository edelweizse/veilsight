#pragma once

#include <encode/mjpeg_server.hpp>
#include <pipeline/publishers.hpp>

namespace veilsight {
    class MJPEGStreamPublisher final : public IStreamPublisher {
    public:
        explicit MJPEGStreamPublisher(MJPEGServer& server) : server_(server) {}

        void register_stream(const std::string& stream_key) override {
            server_.register_stream(stream_key);
        }

        void publish_frame(const std::string& stream_key,
                           const cv::Mat& frame,
                           int quality) override {
            server_.push_jpeg(stream_key, frame, quality);
        }

        void publish_metadata(const std::string& stream_key, std::string json) override {
            server_.push_meta(stream_key, std::move(json));
        }

        void publish_metrics(std::string json) override {
            server_.push_metrics(std::move(json));
        }

    private:
        MJPEGServer& server_;
    };
}
