#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <anonymization/anonymizer.hpp>
#include <common/config.hpp>
#include <encode/mjpeg_server.hpp>
#include <inference/detector.hpp>
#include <inference/recognizer.hpp>
#include <inference/stream_inference_state.hpp>
#include <ingest/gst_dual_source.hpp>
#include <pipeline/bounded_queue.hpp>
#include <pipeline/metrics.hpp>
#include <pipeline/types.hpp>

namespace ss {
    class PipelineRuntime {
    public:
        struct Options {
            int jpeg_quality = 75;

            size_t infer_in_cap = 50;
            size_t inf_state_in_cap = 5;
            size_t det_res_cap = 20;
            size_t recognizer_in_cap = 50;
            size_t recognizer_done_cap = 20;
            size_t anon_in_cap = 50;
            size_t enc_in_cap = 5;
            size_t analytics_cap = 256;

            int64_t reorder_window = 5;
            size_t pending_state_limit = 500;
            int anonymizer_workers = 1;

            DetectorModuleConfig detector;
            TrackerModuleConfig tracker;
            RecognizerModuleConfig recognizer;

            std::string anonymizer_method = "pixelate";
            int anonymizer_pixelation_divisor = 10;
            int anonymizer_blur_kernel = 31;

            MetricsConfig metrics;
        };

        PipelineRuntime(MJPEGServer& server,
                        std::vector<IngestConfig> streams,
                        Options opt);

        bool start();
        void stop();
        bool pop_tracker_output(TrackerFrameOutput& out, std::chrono::milliseconds timeout);

        ~PipelineRuntime();

    private:
        struct StreamPipe {
            std::string stream_id;

            BoundedQueue<FramePtr> inf_state_in;
            BoundedQueue<InferResults> det_res;
            BoundedQueue<FramePtr> recognizer_done;
            BoundedQueue<FramePtr> enc_in;
            std::unique_ptr<StreamInferenceState> inf_state;
            std::map<int64_t, FramePtr> pending_recognized;
            int64_t next_commit_frame_id = -1;
            int64_t latest_queued_recognizer_frame_id = -1;

            std::thread ingest_thr;
            std::thread inf_state_thr;
            std::thread enc_thr;

            StreamPipe(std::string id,
                       size_t inf_state_cap,
                       size_t det_res_queue_cap,
                       size_t recognizer_done_cap,
                       size_t enc_cap)
                : stream_id(std::move(id)),
                  inf_state_in(inf_state_cap),
                  det_res(det_res_queue_cap),
                  recognizer_done(recognizer_done_cap),
                  enc_in(enc_cap) {}
        };

        struct DetectorStage;
        struct RecognizerStage;

        void ingest_loop_(const IngestConfig& cfg, std::unique_ptr<GstDualSource> src, StreamPipe* pipe);
        void infer_state_loop_(StreamPipe* pipe);
        void anonymizer_loop_();
        void encoder_loop_(StreamPipe* pipe);
        void metrics_loop_();

        void enqueue_recognizer_(StreamPipe* pipe, const FramePtr& frame);
        void publish_recognized_frame_(const FramePtr& frame);
        void drain_recognized_ready_(StreamPipe* pipe);
        void commit_frame_(const FramePtr& frame);
        bool validate_thread_budget_(const IDetectorFactory& detector_factory,
                                     const IRecognizerFactory& recognizer_factory) const;

        void anonymize_(cv::Mat& ui_frame,
                        const std::vector<Box>& boxes,
                        float sx,
                        float sy,
                        float tx,
                        float ty);
        void draw_tracks_(cv::Mat& ui_frame,
                          const std::vector<Box>& boxes,
                          float sx,
                          float sy,
                          float tx,
                          float ty);
        void publish_tracker_output_(const FrameCtx& ctx, const std::vector<Box>& tracks);
        std::map<std::string, QueueSnapshot> snapshot_queues_() const;

        MJPEGServer& server_;
        std::vector<IngestConfig> streams_;
        Options opt_;

        std::atomic<bool> running_{false};
        BoundedQueue<FramePtr> anon_in_;
        BoundedQueue<TrackerFrameOutput> analytics_out_;

        std::vector<std::unique_ptr<StreamPipe>> pipes_;
        std::unordered_map<std::string, StreamPipe*> pipes_by_stream_id_;
        std::unique_ptr<DetectorStage> detector_stage_;
        std::unique_ptr<RecognizerStage> recognizer_stage_;
        std::vector<std::thread> anonymizer_pool_;
        std::thread metrics_thr_;

        std::unique_ptr<Anonymizer> anonymizer_;
        std::unique_ptr<RuntimeMetrics> metrics_;
    };
}
