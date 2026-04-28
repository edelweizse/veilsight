#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include <common/config.hpp>

namespace httplib {
    class Response;
}

namespace veilsight {
    class WebRTCPublisher;

    class RunnerHTTPServer {
    public:
        RunnerHTTPServer(std::string host, int port, StreamingConfig streaming);
        ~RunnerHTTPServer();

        bool start();
        void stop();

        void set_webrtc_publisher(WebRTCPublisher* publisher);

        void push_jpeg(const std::string& stream_key,
                       std::shared_ptr<const std::vector<uint8_t>> jpeg);
        void push_jpeg(const std::string& stream_key, const cv::Mat& frame, int quality);
        void push_meta(const std::string& stream_key, std::string json);
        void push_metrics(std::string json);
        void register_stream(const std::string& stream_key);
        std::vector<std::string> list_streams() const;

    private:
        struct Impl;
        struct StreamState {
            mutable std::mutex mtx;
            std::condition_variable cv;
            std::shared_ptr<const std::vector<uint8_t>> last_jpeg;
            uint64_t seq = 0;
            mutable std::mutex meta_mtx;
            std::string last_meta;
        };

        std::shared_ptr<StreamState> get_or_create_(const std::string& stream_key) const;
        std::shared_ptr<StreamState> get_(const std::string& stream_key) const;

        std::string cors_origin_(const std::string& request_origin) const;
        void apply_cors_(const std::string& request_origin, httplib::Response& res) const;

        Impl* impl_ = nullptr;
        std::string host_;
        int port_;
        StreamingConfig streaming_;
        WebRTCPublisher* webrtc_ = nullptr;

        std::thread server_thread_;
        std::atomic<bool> running_{false};

        mutable std::mutex streams_mtx_;
        mutable std::unordered_map<std::string, std::shared_ptr<StreamState>> streams_;

        mutable std::mutex metrics_mtx_;
        std::string last_metrics_json_;
    };
}
