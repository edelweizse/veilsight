#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <common/config.hpp>
#include <opencv2/core.hpp>

namespace veilsight {
    struct H264EncoderSelection {
        bool available = false;
        std::string encoder;
        std::string error;
    };

    class H264EncoderSelector {
    public:
        using AvailabilityFn = std::function<bool(const std::string&)>;

        explicit H264EncoderSelector(AvailabilityFn availability = {});
        H264EncoderSelection select(const StreamingConfig& config) const;

    private:
        AvailabilityFn availability_;
    };

    class WebRTCPublisher {
    public:
        struct OfferResult {
            int status = 503;
            std::string content_type = "text/plain";
            std::string body;
            std::string location;
        };

        explicit WebRTCPublisher(StreamingConfig config);

        bool available() const;
        std::string unavailable_reason() const;
        H264EncoderSelection encoder_selection() const;

        void register_stream(const std::string& stream_key);
        void publish_frame(const std::string& stream_key, const cv::Mat& frame);

        OfferResult handle_offer(const std::string& stream_key, const std::string& sdp_offer);
        bool delete_session(const std::string& session_id);
        void expire_idle_sessions();
        size_t active_sessions(const std::string& stream_key) const;

    private:
        struct Session {
            std::string id;
            std::string stream_key;
            std::chrono::steady_clock::time_point last_seen;
        };

        static std::string make_session_id_();
        static std::string minimal_sdp_answer_();

        StreamingConfig config_;
        H264EncoderSelection encoder_;

        mutable std::mutex mutex_;
        std::map<std::string, Session> sessions_;
        std::vector<std::string> streams_;
    };
}
