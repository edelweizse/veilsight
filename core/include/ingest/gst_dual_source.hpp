#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct _GstElement;
using GstElement = _GstElement;

namespace ss {
    struct FramePacket {
        cv::Mat bgr;
        int64_t pts_ns = 0;
    };

    struct DualFramePacket {
        cv::Mat inf_frame;
        cv::Mat ui_frame;

        int64_t pts_ns = 0;
        int64_t frame_id = 0;

        float scale_x = 0.0f;
        float scale_y = 0.0f;
        float offset_x = 0.0f;
        float offset_y = 0.0f;
    };

    class GstDualSource {
    public:
        GstDualSource(std::string pipeline,
                      std::string id,
                      std::string sink_inf_name,
                      std::string sink_ui_name);

        bool start();
        void stop();

        bool read(DualFramePacket& out, int timeout_ms);

        const std::string& id() const { return id_; }

        ~GstDualSource();

    private:
        struct PendingFrame {
            cv::Mat bgr;
            int64_t pts_ns = 0;
        };

        bool pull_bgr_(GstElement* sink, FramePacket& out, int timeout_ms);
        bool try_match_(DualFramePacket& out);
        bool emit_pair_(PendingFrame inf_pkt, PendingFrame ui_pkt, DualFramePacket& out);
        void trim_pending_();

        std::string pipeline_str_;
        std::string id_;
        std::string sink_inf_name_;
        std::string sink_ui_name_;

        GstElement* pipeline_ = nullptr;
        GstElement* sink_inf_ = nullptr;
        GstElement* sink_ui_ = nullptr;

        int64_t next_frame_id_ = 0;
        size_t pending_cap_ = 8;
        std::deque<PendingFrame> pending_inf_;
        std::deque<PendingFrame> pending_ui_;

        float scale_x_ = 1.0f;
        float scale_y_ = 1.0f;
    };
}
