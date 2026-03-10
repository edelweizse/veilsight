#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <unordered_map>

#include <opencv2/core.hpp>

namespace ss {
    class MJPEGServer {
    public:
        MJPEGServer(std::string host, int port);
        ~MJPEGServer();

        // Start http server in bg thread
        bool start();
        void stop();

        // push latest jpeg frame in bytes
        void push_jpeg(const std::string& stream_key,
                       std::shared_ptr<const std::vector<uint8_t>> jpeg);

        void push_jpeg(const std::string& stream_key,
                      const cv::Mat& frame,
                      int quality);

        // optionally push JSON metadata
        void push_meta(const std::string& stream_key, std::string json);
        void push_metrics(std::string json);

        void register_stream(const std::string& stream_key);

        std::vector<std::string> list_streams() const;

    private:
        struct Impl;
        Impl* impl_ = nullptr;

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

        std::string host_;
        int port_;

        std::thread server_thread_;
        std::atomic<bool> running_{false};

        mutable std::mutex streams_mtx_;
        mutable std::unordered_map<std::string, std::shared_ptr<StreamState>> streams_;

        mutable std::mutex metrics_mtx_;
        std::string last_metrics_json_;
    };
}
