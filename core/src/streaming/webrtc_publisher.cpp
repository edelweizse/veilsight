#include <streaming/webrtc_publisher.hpp>

#include <algorithm>
#include <atomic>
#include <sstream>

#include <gst/gst.h>

namespace veilsight {
    namespace {
        bool gst_factory_available(const std::string& name) {
            if (!gst_is_initialized()) {
                int argc = 0;
                char** argv = nullptr;
                gst_init(&argc, &argv);
            }
            GstElementFactory* factory = gst_element_factory_find(name.c_str());
            if (!factory) return false;
            gst_object_unref(factory);
            return true;
        }

        bool contains_stream(const std::vector<std::string>& streams, const std::string& stream_key) {
            return std::find(streams.begin(), streams.end(), stream_key) != streams.end();
        }
    }

    H264EncoderSelector::H264EncoderSelector(AvailabilityFn availability)
        : availability_(std::move(availability)) {
        if (!availability_) availability_ = gst_factory_available;
    }

    H264EncoderSelection H264EncoderSelector::select(const StreamingConfig& config) const {
        if (!availability_("webrtcbin")) {
            return {false, {}, "GStreamer webrtcbin element is unavailable"};
        }

        if (config.encoder != "auto") {
            if (availability_(config.encoder)) return {true, config.encoder, {}};
            return {false, config.encoder, "configured H.264 encoder is unavailable: " + config.encoder};
        }

        static const std::vector<std::string> kEncoders = {
            "v4l2h264enc",
            "vaapih264enc",
            "nvh264enc",
            "x264enc",
            "openh264enc",
        };
        for (const auto& encoder : kEncoders) {
            if (availability_(encoder)) return {true, encoder, {}};
        }
        return {false, {}, "no supported H.264 encoder is available"};
    }

    WebRTCPublisher::WebRTCPublisher(StreamingConfig config)
        : config_(std::move(config)),
          encoder_(H264EncoderSelector().select(config_)) {}

    bool WebRTCPublisher::available() const {
        return config_.webrtc.enabled && encoder_.available;
    }

    std::string WebRTCPublisher::unavailable_reason() const {
        if (!config_.webrtc.enabled) return "WebRTC is disabled by config";
        return encoder_.error;
    }

    H264EncoderSelection WebRTCPublisher::encoder_selection() const {
        return encoder_;
    }

    void WebRTCPublisher::register_stream(const std::string& stream_key) {
        std::lock_guard lk(mutex_);
        if (!contains_stream(streams_, stream_key)) streams_.push_back(stream_key);
    }

    void WebRTCPublisher::publish_frame(const std::string& stream_key, const cv::Mat& frame) {
        (void)stream_key;
        (void)frame;
        // Real per-peer appsrc fanout belongs here. This method intentionally
        // never blocks the pipeline; slow or incomplete sessions simply miss frames.
    }

    WebRTCPublisher::OfferResult WebRTCPublisher::handle_offer(const std::string& stream_key,
                                                               const std::string& sdp_offer) {
        expire_idle_sessions();

        if (!available()) {
            return {503, "text/plain", unavailable_reason(), {}};
        }
        if (sdp_offer.empty()) {
            return {400, "text/plain", "empty SDP offer", {}};
        }

        std::lock_guard lk(mutex_);
        if (!contains_stream(streams_, stream_key)) {
            return {404, "text/plain", "unknown stream", {}};
        }

        size_t peers = 0;
        for (const auto& [_, session] : sessions_) {
            if (session.stream_key == stream_key) ++peers;
        }
        if (peers >= static_cast<size_t>(config_.webrtc.max_peers_per_stream)) {
            return {429, "text/plain", "too many WebRTC peers for stream", {}};
        }

        const std::string session_id = make_session_id_();
        sessions_[session_id] = Session{session_id, stream_key, std::chrono::steady_clock::now()};
        return {201, "application/sdp", minimal_sdp_answer_(), "/webrtc/whep/sessions/" + session_id};
    }

    bool WebRTCPublisher::delete_session(const std::string& session_id) {
        std::lock_guard lk(mutex_);
        return sessions_.erase(session_id) > 0;
    }

    void WebRTCPublisher::expire_idle_sessions() {
        const auto now = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::seconds(config_.webrtc.session_idle_timeout_s);
        std::lock_guard lk(mutex_);
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            if (now - it->second.last_seen > timeout) it = sessions_.erase(it);
            else ++it;
        }
    }

    size_t WebRTCPublisher::active_sessions(const std::string& stream_key) const {
        std::lock_guard lk(mutex_);
        size_t peers = 0;
        for (const auto& [_, session] : sessions_) {
            if (session.stream_key == stream_key) ++peers;
        }
        return peers;
    }

    std::string WebRTCPublisher::make_session_id_() {
        static std::atomic<uint64_t> next_id{1};
        std::ostringstream oss;
        oss << "sess-" << next_id.fetch_add(1, std::memory_order_relaxed);
        return oss.str();
    }

    std::string WebRTCPublisher::minimal_sdp_answer_() {
        return "v=0\r\n"
               "o=veilsight 0 0 IN IP4 127.0.0.1\r\n"
               "s=Veilsight\r\n"
               "t=0 0\r\n"
               "a=group:BUNDLE 0\r\n"
               "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
               "c=IN IP4 0.0.0.0\r\n"
               "a=mid:0\r\n"
               "a=recvonly\r\n"
               "a=rtpmap:96 H264/90000\r\n";
    }
}
