#include <ingest/gst_dual_source.hpp>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <utility>

namespace ss {
    GstDualSource::GstDualSource(std::string pipeline,
                                 std::string id,
                                 std::string sink_inf_name,
                                 std::string sink_ui_name)
        : pipeline_str_(std::move(pipeline)),
          id_(std::move(id)),
          sink_inf_name_(std::move(sink_inf_name)),
          sink_ui_name_(std::move(sink_ui_name)) {}

    bool GstDualSource::start() {
        static std::once_flag gst_init_flag;
        std::call_once(gst_init_flag, [] { gst_init(nullptr, nullptr); });

        pending_inf_.clear();
        pending_ui_.clear();
        next_frame_id_ = 0;
        scale_x_ = 1.0f;
        scale_y_ = 1.0f;

        GError* err = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str_.c_str(), &err);
        if (!pipeline_) {
            if (err) {
                std::cerr << "[GStreamer](start) parse_launch error: " << err->message << "\n";
                g_error_free(err);
            } else {
                std::cerr << "[GStreamer](start) parse_launch failed (unk error)\n";
            }
            return false;
        }

        sink_inf_ = gst_bin_get_by_name(GST_BIN(pipeline_), sink_inf_name_.c_str());
        sink_ui_ = gst_bin_get_by_name(GST_BIN(pipeline_), sink_ui_name_.c_str());

        if (!sink_inf_ || !sink_ui_) {
            std::cerr << "[GStreamer](start) missing appsink(s): "
                      << sink_inf_name_ << " and/or " << sink_ui_name_ << "\n";
            stop();
            return false;
        }

        for (auto* s : {sink_inf_, sink_ui_}) {
            auto* appsink = GST_APP_SINK(s);
            gst_app_sink_set_drop(appsink, TRUE);
            gst_app_sink_set_max_buffers(appsink, 1);
            gst_app_sink_set_emit_signals(appsink, FALSE);
        }

        const GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "[GStreamer](start) Failed to set pipeline to PLAYING.\n";
            stop();
            return false;
        }
        return true;
    }

    bool GstDualSource::pull_bgr_(GstElement* sink, FramePacket& out, int timeout_ms) {
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), timeout_ms * GST_MSECOND);
        if (!sample) return false;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        if (!buffer || !caps) {
            gst_sample_unref(sample);
            return false;
        }

        GstStructure* st = gst_caps_get_structure(caps, 0);
        int width = 0;
        int height = 0;
        gst_structure_get_int(st, "width", &width);
        gst_structure_get_int(st, "height", &height);
        if (width <= 0 || height <= 0) {
            gst_sample_unref(sample);
            return false;
        }

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ) || !map.data || map.size == 0) {
            gst_sample_unref(sample);
            return false;
        }

        GstVideoInfo vinfo;
        int stride = width * 3;
        if (gst_video_info_from_caps(&vinfo, caps)) {
            const int s0 = GST_VIDEO_INFO_PLANE_STRIDE(&vinfo, 0);
            if (s0 > 0) stride = s0;
        }

        const size_t min_bytes = static_cast<size_t>(stride) * static_cast<size_t>(height);
        if (map.size < min_bytes) {
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            return false;
        }

        cv::Mat tmp(height, width, CV_8UC3, map.data, stride);
        out.bgr = tmp.clone();
        out.pts_ns = (buffer->pts == GST_CLOCK_TIME_NONE) ? 0 : static_cast<int64_t>(buffer->pts);

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return true;
    }

    bool GstDualSource::emit_pair_(PendingFrame inf_pkt, PendingFrame ui_pkt, DualFramePacket& out) {
        if (inf_pkt.bgr.empty() || ui_pkt.bgr.empty()) return false;

        scale_x_ = static_cast<float>(ui_pkt.bgr.cols) / static_cast<float>(std::max(1, inf_pkt.bgr.cols));
        scale_y_ = static_cast<float>(ui_pkt.bgr.rows) / static_cast<float>(std::max(1, inf_pkt.bgr.rows));

        out.inf_frame = std::move(inf_pkt.bgr);
        out.ui_frame = std::move(ui_pkt.bgr);
        out.pts_ns = (inf_pkt.pts_ns > 0) ? inf_pkt.pts_ns : ui_pkt.pts_ns;
        out.frame_id = next_frame_id_++;
        out.scale_x = scale_x_;
        out.scale_y = scale_y_;
        out.offset_x = 0.0f;
        out.offset_y = 0.0f;
        return true;
    }

    bool GstDualSource::try_match_(DualFramePacket& out) {
        for (auto inf_it = pending_inf_.begin(); inf_it != pending_inf_.end(); ++inf_it) {
            if (inf_it->pts_ns <= 0) continue;
            const auto ui_it = std::find_if(pending_ui_.begin(), pending_ui_.end(), [pts = inf_it->pts_ns](const PendingFrame& pkt) {
                return pkt.pts_ns == pts;
            });
            if (ui_it == pending_ui_.end()) continue;

            PendingFrame inf_pkt = std::move(*inf_it);
            PendingFrame ui_pkt = std::move(*ui_it);
            pending_inf_.erase(inf_it);
            pending_ui_.erase(ui_it);
            return emit_pair_(std::move(inf_pkt), std::move(ui_pkt), out);
        }

        if (!pending_inf_.empty() && !pending_ui_.empty() &&
            pending_inf_.front().pts_ns <= 0 && pending_ui_.front().pts_ns <= 0) {
            PendingFrame inf_pkt = std::move(pending_inf_.front());
            PendingFrame ui_pkt = std::move(pending_ui_.front());
            pending_inf_.pop_front();
            pending_ui_.pop_front();
            return emit_pair_(std::move(inf_pkt), std::move(ui_pkt), out);
        }

        return false;
    }

    void GstDualSource::trim_pending_() {
        while (pending_inf_.size() > pending_cap_) {
            pending_inf_.pop_front();
        }
        while (pending_ui_.size() > pending_cap_) {
            pending_ui_.pop_front();
        }

        while (!pending_inf_.empty() && !pending_ui_.empty()) {
            const int64_t inf_pts = pending_inf_.front().pts_ns;
            const int64_t ui_pts = pending_ui_.front().pts_ns;
            if (inf_pts <= 0 || ui_pts <= 0 || inf_pts == ui_pts) break;

            const bool inf_has_match = std::any_of(pending_ui_.begin(), pending_ui_.end(), [inf_pts](const PendingFrame& pkt) {
                return pkt.pts_ns == inf_pts;
            });
            const bool ui_has_match = std::any_of(pending_inf_.begin(), pending_inf_.end(), [ui_pts](const PendingFrame& pkt) {
                return pkt.pts_ns == ui_pts;
            });

            if (!inf_has_match && (ui_has_match || inf_pts < ui_pts)) {
                pending_inf_.pop_front();
                continue;
            }
            if (!ui_has_match && (inf_has_match || ui_pts < inf_pts)) {
                pending_ui_.pop_front();
                continue;
            }
            break;
        }
    }

    bool GstDualSource::read(DualFramePacket& out, int timeout_ms) {
        if (!sink_inf_ || !sink_ui_) return false;

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(0, timeout_ms));
        while (std::chrono::steady_clock::now() < deadline) {
            if (try_match_(out)) return true;

            const auto now = std::chrono::steady_clock::now();
            const int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            if (remaining_ms <= 0) break;

            const int poll_ms = std::max(1, std::min(remaining_ms, 10));
            bool pulled_any = false;

            FramePacket inf_pkt;
            if (pull_bgr_(sink_inf_, inf_pkt, poll_ms)) {
                pending_inf_.push_back(PendingFrame{std::move(inf_pkt.bgr), inf_pkt.pts_ns});
                pulled_any = true;
            }
            if (try_match_(out)) return true;

            FramePacket ui_pkt;
            if (pull_bgr_(sink_ui_, ui_pkt, poll_ms)) {
                pending_ui_.push_back(PendingFrame{std::move(ui_pkt.bgr), ui_pkt.pts_ns});
                pulled_any = true;
            }

            trim_pending_();
            if (!pulled_any && remaining_ms <= poll_ms) break;
        }

        trim_pending_();
        return try_match_(out);
    }

    void GstDualSource::stop() {
        pending_inf_.clear();
        pending_ui_.clear();
        next_frame_id_ = 0;
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);

            if (sink_inf_) {
                gst_object_unref(sink_inf_);
                sink_inf_ = nullptr;
            }
            if (sink_ui_) {
                gst_object_unref(sink_ui_);
                sink_ui_ = nullptr;
            }

            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

    GstDualSource::~GstDualSource() {
        stop();
    }
}
