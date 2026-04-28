#pragma once

#include <pipeline/publishers.hpp>
#include <streaming/mjpeg_publisher.hpp>
#include <streaming/webrtc_publisher.hpp>

namespace veilsight {
    class CompositeStreamPublisher final : public IStreamPublisher {
    public:
        CompositeStreamPublisher(MJPEGPublisher& fallback,
                                 WebRTCPublisher& webrtc,
                                 bool mjpeg_fallback_enabled)
            : fallback_(fallback),
              webrtc_(webrtc),
              mjpeg_fallback_enabled_(mjpeg_fallback_enabled) {}

        void register_stream(const std::string& stream_key) override {
            fallback_.register_stream(stream_key);
            webrtc_.register_stream(stream_key);
        }

        void publish_frame(const std::string& stream_key,
                           const cv::Mat& frame,
                           int quality) override {
            webrtc_.publish_frame(stream_key, frame);
            if (mjpeg_fallback_enabled_) fallback_.publish_frame(stream_key, frame, quality);
        }

        void publish_metadata(const std::string& stream_key, std::string json) override {
            fallback_.publish_metadata(stream_key, std::move(json));
        }

        void publish_metrics(std::string json) override {
            fallback_.publish_metrics(std::move(json));
        }

    private:
        MJPEGPublisher& fallback_;
        WebRTCPublisher& webrtc_;
        bool mjpeg_fallback_enabled_;
    };
}
