#include <pipeline/runtime.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "ingest/dual_source_factory.hpp"

namespace veilsight {
    namespace {
        uint64_t steady_now_ns() {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
        }

        Box map_box_to_ui(const Box& box, const FrameCtx& frame) {
            Box out = box;
            out.x = box.x * frame.scale_x + frame.offset_x;
            out.y = box.y * frame.scale_y + frame.offset_y;
            out.w = box.w * frame.scale_x;
            out.h = box.h * frame.scale_y;
            return out;
        }
    }

    struct PipelineRuntime::DetectorStage {
        explicit DetectorStage(size_t input_cap, std::unique_ptr<IDetectorFactory> stage_factory)
            : input(input_cap),
              factory(std::move(stage_factory)) {}

        BoundedQueue<FramePtr> input;
        std::unique_ptr<IDetectorFactory> factory;
        std::vector<std::thread> workers;
    };

    struct PipelineRuntime::RecognizerStage {
        explicit RecognizerStage(size_t input_cap, std::unique_ptr<IRecognizerFactory> stage_factory)
            : input(input_cap),
              factory(std::move(stage_factory)) {}

        BoundedQueue<FramePtr> input;
        std::unique_ptr<IRecognizerFactory> factory;
        std::vector<std::thread> workers;
    };

    PipelineRuntime::PipelineRuntime(IStreamPublisher& stream_publisher,
                                     ITelemetryPublisher& telemetry_publisher,
                                     std::vector<IngestConfig> streams,
                                     Options opt)
        : stream_publisher_(stream_publisher),
          telemetry_publisher_(telemetry_publisher),
          streams_(std::move(streams)),
          opt_(opt),
          anon_in_(opt.anon_in_cap),
          analytics_out_(opt.analytics_cap) {}

    PipelineRuntime::~PipelineRuntime() {
        stop();
    }

    bool PipelineRuntime::start() {
        if (running_) return true;

        auto detector_factory = create_detector_factory(opt_.detector);
        auto tracker_factory = create_tracker_factory(opt_.tracker);
        auto recognizer_factory = create_recognizer_factory(opt_.recognizer);
        if (!detector_factory || !tracker_factory || !recognizer_factory) {
            std::cerr << "[Pipeline](start) failed to create stage factories.\n";
            return false;
        }
        if (!validate_thread_budget_(*detector_factory, *recognizer_factory)) {
            return false;
        }

        anon_in_.reset();
        analytics_out_.reset();
        if (opt_.metrics.enabled) {
            metrics_ = std::make_unique<RuntimeMetrics>();
        } else {
            metrics_.reset();
        }

        try {
            AnonymizerConfig acfg;
            acfg.method = opt_.anonymizer_method;
            acfg.pixelation_divisor = opt_.anonymizer_pixelation_divisor;
            acfg.blur_kernel = opt_.anonymizer_blur_kernel;
            anonymizer_ = std::make_unique<Anonymizer>(std::move(acfg));
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline](start) anonymizer init failed: " << e.what() << "\n";
            anonymizer_.reset();
            metrics_.reset();
            return false;
        }

        pipes_.clear();
        pipes_by_stream_id_.clear();
        pipes_.reserve(streams_.size());
        for (const auto& s : streams_) {
            auto pipe = std::make_unique<StreamPipe>(s.id,
                                                     opt_.inf_state_in_cap,
                                                     opt_.det_res_cap,
                                                     opt_.recognizer_done_cap,
                                                     opt_.enc_in_cap);
            try {
                pipe->inf_state = std::make_unique<StreamInferenceState>(
                    tracker_factory->create(),
                    opt_.reorder_window,
                    opt_.pending_state_limit);
            } catch (const std::exception& e) {
                std::cerr << "[Pipeline](start) tracker init failed for " << s.id << ": " << e.what() << "\n";
                continue;
            }
            pipes_by_stream_id_[s.id] = pipe.get();
            pipes_.push_back(std::move(pipe));
        }

        detector_stage_ = std::make_unique<DetectorStage>(opt_.infer_in_cap, std::move(detector_factory));
        recognizer_stage_ = std::make_unique<RecognizerStage>(opt_.recognizer_in_cap, std::move(recognizer_factory));

        running_ = true;
        try {
            detector_stage_->workers.clear();
            detector_stage_->workers.reserve(std::max(1, opt_.detector.workers));
            for (int i = 0; i < std::max(1, opt_.detector.workers); ++i) {
                auto detector = detector_stage_->factory->create();
                detector_stage_->workers.emplace_back([this, detector = std::move(detector)]() mutable {
                    while (running_.load(std::memory_order_relaxed)) {
                        FramePtr ctx;
                        if (!detector_stage_->input.pop_for(ctx, std::chrono::milliseconds(200))) continue;
                        if (!ctx || !detector) continue;

                        InferResults res;
                        res.stream_id = ctx->stream_id;
                        res.frame_id = ctx->frame_id;
                        bool ok = true;
                        const uint64_t detector_t0_ns = steady_now_ns();

                        try {
                            res.bboxes = detector->detect(ctx->inf);
                        } catch (const std::exception& e) {
                            ok = false;
                            thread_local bool logged = false;
                            if (!logged) {
                                std::cerr << "[Pipeline](detector) detect failed: " << e.what() << "\n";
                                logged = true;
                            }
                            res.bboxes.clear();
                        }

                        if (metrics_) {
                            const uint64_t dt_ns = steady_now_ns() - detector_t0_ns;
                            metrics_->observe_global(RuntimeStage::Detector, dt_ns, ok);
                            metrics_->observe_stream(ctx->stream_id, RuntimeStage::Detector, dt_ns, ok);
                        }

                        auto it = pipes_by_stream_id_.find(ctx->stream_id);
                        if (it != pipes_by_stream_id_.end() && it->second) {
                            it->second->det_res.push_drop_oldest(std::move(res));
                        }
                    }
                });
            }

            recognizer_stage_->workers.clear();
            recognizer_stage_->workers.reserve(std::max(1, opt_.recognizer.workers));
            for (int i = 0; i < std::max(1, opt_.recognizer.workers); ++i) {
                auto recognizer = recognizer_stage_->factory->create();
                recognizer_stage_->workers.emplace_back([this, recognizer = std::move(recognizer)]() mutable {
                    while (running_.load(std::memory_order_relaxed)) {
                        FramePtr ctx;
                        if (!recognizer_stage_->input.pop_for(ctx, std::chrono::milliseconds(200))) continue;
                        if (!ctx || !recognizer) continue;

                        bool ok = true;
                        const uint64_t recognizer_t0_ns = steady_now_ns();
                        try {
                            recognizer->annotate(*ctx, ctx->tracked_boxes);
                        } catch (const std::exception& e) {
                            ok = false;
                            thread_local bool logged = false;
                            if (!logged) {
                                std::cerr << "[Pipeline](recognizer) annotate failed: " << e.what() << "\n";
                                logged = true;
                            }
                        }

                        if (metrics_) {
                            const uint64_t dt_ns = steady_now_ns() - recognizer_t0_ns;
                            metrics_->observe_global(RuntimeStage::Recognizer, dt_ns, ok);
                            metrics_->observe_stream(ctx->stream_id, RuntimeStage::Recognizer, dt_ns, ok);
                        }
                        publish_recognized_frame_(ctx);
                    }
                });
            }
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline](start) stage worker init failed: " << e.what() << "\n";
            stop();
            return false;
        }

        anonymizer_pool_.clear();
        anonymizer_pool_.reserve(std::max(1, opt_.anonymizer_workers));
        for (int i = 0; i < std::max(1, opt_.anonymizer_workers); ++i) {
            anonymizer_pool_.emplace_back([this] { anonymizer_loop_(); });
        }

        size_t started_streams = 0;
        for (const auto& cfg : streams_) {
            auto it = pipes_by_stream_id_.find(cfg.id);
            if (it == pipes_by_stream_id_.end() || !it->second) continue;
            StreamPipe* pipe = it->second;

            std::unique_ptr<GstDualSource> src;
            try {
                src = make_dual_source(cfg);
            } catch (const std::exception& e) {
                std::cerr << "[Pipeline](start) make_dual_source failed for " << cfg.id << ": " << e.what() << "\n";
                continue;
            }

            pipe->ingest_thr = std::thread([this, cfg, pipe, src = std::move(src)]() mutable {
                ingest_loop_(cfg, std::move(src), pipe);
            });
            pipe->inf_state_thr = std::thread([this, pipe] { infer_state_loop_(pipe); });
            pipe->enc_thr = std::thread([this, pipe] { encoder_loop_(pipe); });
            ++started_streams;
        }

        if (started_streams == 0) {
            std::cerr << "[Pipeline](start) no streams were started.\n";
            stop();
            return false;
        }

        if (metrics_) {
            metrics_thr_ = std::thread([this] { metrics_loop_(); });
        }
        return true;
    }

    void PipelineRuntime::stop() {
        if (!running_ && !detector_stage_ && !recognizer_stage_ && pipes_.empty()) return;
        running_ = false;

        anon_in_.stop();
        analytics_out_.stop();
        if (detector_stage_) detector_stage_->input.stop();
        if (recognizer_stage_) recognizer_stage_->input.stop();
        for (auto& pipe : pipes_) {
            pipe->inf_state_in.stop();
            pipe->det_res.stop();
            pipe->recognizer_done.stop();
            pipe->enc_in.stop();
        }

        for (auto& pipe : pipes_) {
            if (pipe->ingest_thr.joinable()) pipe->ingest_thr.join();
            if (pipe->inf_state_thr.joinable()) pipe->inf_state_thr.join();
            if (pipe->enc_thr.joinable()) pipe->enc_thr.join();
        }

        if (detector_stage_) {
            for (auto& worker : detector_stage_->workers) {
                if (worker.joinable()) worker.join();
            }
        }
        if (recognizer_stage_) {
            for (auto& worker : recognizer_stage_->workers) {
                if (worker.joinable()) worker.join();
            }
        }
        for (auto& worker : anonymizer_pool_) {
            if (worker.joinable()) worker.join();
        }
        anonymizer_pool_.clear();
        if (metrics_thr_.joinable()) metrics_thr_.join();

        pipes_.clear();
        pipes_by_stream_id_.clear();
        detector_stage_.reset();
        recognizer_stage_.reset();
        anonymizer_.reset();
        metrics_.reset();
    }

    bool PipelineRuntime::is_running() const {
        return running_.load(std::memory_order_relaxed);
    }

    bool PipelineRuntime::pop_tracker_output(TrackerFrameOutput& out, std::chrono::milliseconds timeout) {
        return analytics_out_.pop_for(out, timeout);
    }

    void PipelineRuntime::ingest_loop_(const IngestConfig& cfg,
                                       std::unique_ptr<GstDualSource> src,
                                       StreamPipe* pipe) {
        if (!src || !pipe || !detector_stage_) return;
        if (!src->start()) {
            std::cerr << "[Pipeline](ingest_loop_) start() failed for " << cfg.id << ".\n";
            return;
        }

        DualFramePacket dp;
        while (running_.load(std::memory_order_relaxed)) {
            if (!src->read(dp, 100)) continue;
            const uint64_t ingest_t0_ns = steady_now_ns();

            auto ctx = std::make_shared<FrameCtx>();
            ctx->stream_id = cfg.id;
            ctx->frame_id = dp.frame_id;
            ctx->pts_ns = dp.pts_ns;
            ctx->created_steady_ns = ingest_t0_ns;
            ctx->scale_x = dp.scale_x;
            ctx->scale_y = dp.scale_y;
            ctx->offset_x = dp.offset_x;
            ctx->offset_y = dp.offset_y;
            ctx->ui = std::move(dp.ui_frame);
            ctx->inf = std::move(dp.inf_frame);
            ctx->inf_w = ctx->inf.cols;
            ctx->inf_h = ctx->inf.rows;
            ctx->ui_w = ctx->ui.cols;
            ctx->ui_h = ctx->ui.rows;

            detector_stage_->input.push_drop_oldest(ctx);
            pipe->inf_state_in.push_drop_oldest(ctx);

            if (metrics_) {
                const uint64_t dt_ns = steady_now_ns() - ingest_t0_ns;
                metrics_->observe_global(RuntimeStage::Ingest, dt_ns);
                metrics_->observe_stream(cfg.id, RuntimeStage::Ingest, dt_ns);
            }
        }
        src->stop();
    }

    void PipelineRuntime::infer_state_loop_(StreamPipe* pipe) {
        if (!pipe || !pipe->inf_state) return;

        StreamInferenceState::Callbacks callbacks;
        callbacks.on_tracker_timing = [this](const FrameCtx& frame, uint64_t duration_ns) {
            if (!metrics_) return;
            metrics_->observe_global(RuntimeStage::Tracker, duration_ns);
            metrics_->observe_stream(frame.stream_id, RuntimeStage::Tracker, duration_ns);
        };
        callbacks.on_frame_ready = [this, pipe](const FramePtr& frame) {
            enqueue_recognizer_(pipe, frame);
        };

        while (running_.load(std::memory_order_relaxed)) {
            FramePtr ctx;
            if (pipe->inf_state_in.pop_for(ctx, std::chrono::milliseconds(2)) && ctx) {
                pipe->inf_state->push_frame(ctx);
            }
            while (pipe->inf_state_in.try_pop(ctx)) {
                if (!ctx) continue;
                pipe->inf_state->push_frame(ctx);
            }

            InferResults det;
            while (pipe->det_res.try_pop(det)) {
                pipe->inf_state->push_detection(std::move(det));
            }

            pipe->inf_state->drain_ready(callbacks);

            FramePtr recognized;
            while (pipe->recognizer_done.try_pop(recognized)) {
                if (!recognized) continue;
                pipe->pending_recognized[recognized->frame_id] = recognized;
            }
            drain_recognized_ready_(pipe);
        }
    }

    void PipelineRuntime::enqueue_recognizer_(StreamPipe* pipe, const FramePtr& frame) {
        if (!pipe || !frame || !recognizer_stage_) return;
        if (pipe->next_commit_frame_id < 0) {
            pipe->next_commit_frame_id = frame->frame_id;
        }
        pipe->latest_queued_recognizer_frame_id = std::max(pipe->latest_queued_recognizer_frame_id,
                                                           frame->frame_id);
        recognizer_stage_->input.push_drop_oldest(frame);
    }

    void PipelineRuntime::publish_recognized_frame_(const FramePtr& frame) {
        if (!frame) return;
        auto it = pipes_by_stream_id_.find(frame->stream_id);
        if (it == pipes_by_stream_id_.end() || !it->second) return;
        it->second->recognizer_done.push_drop_oldest(frame);
    }

    void PipelineRuntime::drain_recognized_ready_(StreamPipe* pipe) {
        if (!pipe) return;
        if (pipe->next_commit_frame_id < 0 && !pipe->pending_recognized.empty()) {
            pipe->next_commit_frame_id = pipe->pending_recognized.begin()->first;
        }

        for (auto it = pipe->pending_recognized.begin(); it != pipe->pending_recognized.end();) {
            if (it->first >= pipe->next_commit_frame_id) break;
            it = pipe->pending_recognized.erase(it);
        }

        while (pipe->next_commit_frame_id >= 0) {
            auto it = pipe->pending_recognized.find(pipe->next_commit_frame_id);
            if (it != pipe->pending_recognized.end()) {
                commit_frame_(it->second);
                pipe->pending_recognized.erase(it);
                ++pipe->next_commit_frame_id;
                continue;
            }

            if (pipe->latest_queued_recognizer_frame_id >= 0 &&
                pipe->latest_queued_recognizer_frame_id - pipe->next_commit_frame_id > opt_.reorder_window) {
                ++pipe->next_commit_frame_id;
                continue;
            }
            break;
        }

        while (pipe->pending_recognized.size() > opt_.pending_state_limit) {
            pipe->pending_recognized.erase(pipe->pending_recognized.begin());
        }
    }

    void PipelineRuntime::commit_frame_(const FramePtr& frame) {
        if (!frame) return;
        publish_tracker_output_(*frame, frame->tracked_boxes);
        frame->inf.release();
        anon_in_.push_drop_oldest(frame);
    }

    void PipelineRuntime::anonymizer_loop_() {
        while (running_.load(std::memory_order_relaxed)) {
            FramePtr ctx;
            if (!anon_in_.pop_for(ctx, std::chrono::milliseconds(200))) continue;
            if (!ctx) continue;
            const uint64_t anonymizer_t0_ns = steady_now_ns();

            anonymize_(ctx->ui,
                       ctx->tracked_boxes,
                       ctx->scale_x,
                       ctx->scale_y,
                       ctx->offset_x,
                       ctx->offset_y);

            if (metrics_) {
                const uint64_t dt_ns = steady_now_ns() - anonymizer_t0_ns;
                metrics_->observe_global(RuntimeStage::Anonymizer, dt_ns);
                metrics_->observe_stream(ctx->stream_id, RuntimeStage::Anonymizer, dt_ns);
            }

            auto it = pipes_by_stream_id_.find(ctx->stream_id);
            if (it != pipes_by_stream_id_.end() && it->second) {
                it->second->enc_in.push_drop_oldest(ctx);
            }
        }
    }

    void PipelineRuntime::encoder_loop_(StreamPipe* pipe) {
        if (!pipe) return;
        const std::string ui_key = pipe->stream_id + "/ui";

        while (running_.load(std::memory_order_relaxed)) {
            FramePtr ctx;
            if (!pipe->enc_in.pop_for(ctx, std::chrono::milliseconds(200))) continue;
            if (!ctx || ctx->ui.empty()) continue;
            const uint64_t encoder_t0_ns = steady_now_ns();

            stream_publisher_.publish_frame(ui_key, ctx->ui, opt_.jpeg_quality);

            std::string ui_meta =
                "{"
                "\"stream_id\":\"" + ctx->stream_id + "\"," 
                "\"profile\":\"ui\"," 
                "\"frame_id\":" + std::to_string(ctx->frame_id) + ","
                "\"pts_ns\":" + std::to_string(ctx->pts_ns) + ","
                "\"w\":" + std::to_string(ctx->ui.cols) + ","
                "\"h\":" + std::to_string(ctx->ui.rows) + ","
                "\"tracks\":" + std::to_string(ctx->tracked_boxes.size()) +
                "}";
            stream_publisher_.publish_metadata(ui_key, std::move(ui_meta));

            if (metrics_) {
                const uint64_t encoder_t1_ns = steady_now_ns();
                const uint64_t encoder_dt_ns = encoder_t1_ns - encoder_t0_ns;
                metrics_->observe_global(RuntimeStage::Encoder, encoder_dt_ns);
                metrics_->observe_stream(ctx->stream_id, RuntimeStage::Encoder, encoder_dt_ns);
                if (ctx->created_steady_ns > 0 && encoder_t1_ns >= ctx->created_steady_ns) {
                    const uint64_t e2e_ns = encoder_t1_ns - ctx->created_steady_ns;
                    metrics_->observe_global(RuntimeStage::EndToEnd, e2e_ns);
                    metrics_->observe_stream(ctx->stream_id, RuntimeStage::EndToEnd, e2e_ns);
                }
            }
        }
    }

    bool PipelineRuntime::validate_thread_budget_(const IDetectorFactory& detector_factory,
                                                  const IRecognizerFactory& recognizer_factory) const {
        const size_t cpu_budget = static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency()));
        const size_t stream_threads = streams_.size() * 3u;
        const size_t detector_parallelism =
            static_cast<size_t>(std::max(1, opt_.detector.workers)) *
            static_cast<size_t>(std::max(1, detector_factory.backend_threads()));
        const size_t recognizer_parallelism =
            static_cast<size_t>(std::max(1, opt_.recognizer.workers)) *
            static_cast<size_t>(std::max(1, recognizer_factory.backend_threads()));
        const size_t anonymizer_threads = static_cast<size_t>(std::max(1, opt_.anonymizer_workers));
        const size_t metrics_threads = opt_.metrics.enabled ? 1u : 0u;
        const size_t total_parallelism =
            stream_threads + detector_parallelism + recognizer_parallelism + anonymizer_threads + metrics_threads;

        if (total_parallelism <= cpu_budget) {
            return true;
        }

        std::cerr << "[Pipeline](start) refusing oversubscribed configuration: required_threads="
                  << total_parallelism
                  << " cpu_budget=" << cpu_budget
                  << " stream_threads=" << stream_threads
                  << " detector_parallelism=" << detector_parallelism
                  << " recognizer_parallelism=" << recognizer_parallelism
                  << " anonymizer_threads=" << anonymizer_threads
                  << " metrics_threads=" << metrics_threads
                  << "\n";
        return false;
    }

    void PipelineRuntime::anonymize_(cv::Mat& ui,
                                     const std::vector<Box>& bboxes,
                                     float sx,
                                     float sy,
                                     float tx,
                                     float ty) {
        if (!anonymizer_) return;
        anonymizer_->apply(ui, bboxes, sx, sy, tx, ty);
    }

    void PipelineRuntime::draw_tracks_(cv::Mat& ui,
                                       const std::vector<Box>& boxes,
                                       float sx,
                                       float sy,
                                       float tx,
                                       float ty) {
        if (ui.empty()) return;

        for (const auto& b : boxes) {
            const int x = static_cast<int>(std::lround(b.x * sx + tx));
            const int y = static_cast<int>(std::lround(b.y * sy + ty));
            const int w = static_cast<int>(std::lround(b.w * sx));
            const int h = static_cast<int>(std::lround(b.h * sy));

            cv::Rect r(x, y, w, h);
            cv::Rect bounds(0, 0, ui.cols, ui.rows);
            r &= bounds;
            if (r.width < 2 || r.height < 2) continue;

            const cv::Scalar color = b.occluded ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(ui, r, color, 2);

            const std::string label = "id:" + std::to_string(b.id);
            const int text_y = std::max(14, r.y - 4);
            cv::putText(ui,
                        label,
                        cv::Point(r.x, text_y),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        1,
                        cv::LINE_AA);
        }
    }

    void PipelineRuntime::publish_tracker_output_(const FrameCtx& ctx, const std::vector<Box>& tracks) {
        std::vector<Box> ui_tracks;
        ui_tracks.reserve(tracks.size());
        for (const auto& track : tracks) {
            ui_tracks.push_back(map_box_to_ui(track, ctx));
        }

        TrackerFrameOutput out;
        out.stream_id = ctx.stream_id;
        out.frame_id = ctx.frame_id;
        out.pts_ns = ctx.pts_ns;
        out.tracks = ui_tracks;
        out.width = ctx.ui_w;
        out.height = ctx.ui_h;
        analytics_out_.push_drop_oldest(std::move(out));
        telemetry_publisher_.publish_frame_analytics(ctx, ui_tracks);
    }

    std::map<std::string, QueueSnapshot> PipelineRuntime::snapshot_queues_() const {
        std::map<std::string, QueueSnapshot> out;
        if (detector_stage_) {
            out["infer_in"] = QueueSnapshot{detector_stage_->input.size(),
                                             detector_stage_->input.capacity(),
                                             detector_stage_->input.dropped_count()};
        }
        if (recognizer_stage_) {
            out["recognizer_in"] = QueueSnapshot{recognizer_stage_->input.size(),
                                                  recognizer_stage_->input.capacity(),
                                                  recognizer_stage_->input.dropped_count()};
        }
        out["anon_in"] = QueueSnapshot{anon_in_.size(), anon_in_.capacity(), anon_in_.dropped_count()};
        out["analytics_out"] = QueueSnapshot{analytics_out_.size(), analytics_out_.capacity(), analytics_out_.dropped_count()};

        for (const auto& pipe : pipes_) {
            if (!pipe) continue;
            out[pipe->stream_id + "/inf_state_in"] =
                QueueSnapshot{pipe->inf_state_in.size(), pipe->inf_state_in.capacity(), pipe->inf_state_in.dropped_count()};
            out[pipe->stream_id + "/det_res"] =
                QueueSnapshot{pipe->det_res.size(), pipe->det_res.capacity(), pipe->det_res.dropped_count()};
            out[pipe->stream_id + "/recognizer_done"] =
                QueueSnapshot{pipe->recognizer_done.size(), pipe->recognizer_done.capacity(), pipe->recognizer_done.dropped_count()};
            out[pipe->stream_id + "/enc_in"] =
                QueueSnapshot{pipe->enc_in.size(), pipe->enc_in.capacity(), pipe->enc_in.dropped_count()};
        }
        return out;
    }

    void PipelineRuntime::metrics_loop_() {
        if (!metrics_) return;

        const auto tick = std::chrono::milliseconds(250);
        auto next_log = std::chrono::steady_clock::now();

        while (running_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(tick);
            if (!running_.load(std::memory_order_relaxed)) break;

            const RuntimeMetrics::Snapshot snap = metrics_->snapshot();
            const auto queues = snapshot_queues_();
            const std::string json = metrics_snapshot_to_json(snap, queues);

            if (opt_.metrics.enable_http || opt_.metrics.enable_ui_payload) {
                stream_publisher_.publish_metrics(json);
            }
            telemetry_publisher_.publish_metrics_snapshot(snap, queues);

            const auto now = std::chrono::steady_clock::now();
            if (now < next_log) continue;
            next_log = now + std::chrono::milliseconds(opt_.metrics.log_interval_ms);

            const auto pick_stage = [&snap](RuntimeStage stage) -> StageSnapshot {
                auto it = snap.global.find(stage);
                return (it != snap.global.end()) ? it->second : StageSnapshot{};
            };
            const StageSnapshot det = pick_stage(RuntimeStage::Detector);
            const StageSnapshot trk = pick_stage(RuntimeStage::Tracker);
            const StageSnapshot rec = pick_stage(RuntimeStage::Recognizer);
            const StageSnapshot ano = pick_stage(RuntimeStage::Anonymizer);
            const StageSnapshot enc = pick_stage(RuntimeStage::Encoder);
            const StageSnapshot e2e = pick_stage(RuntimeStage::EndToEnd);

            std::cerr << "[Metrics] "
                      << "det_fps=" << det.fps << " det_p95_ms=" << det.p95_ms << " "
                      << "trk_fps=" << trk.fps << " trk_p95_ms=" << trk.p95_ms << " "
                      << "rec_fps=" << rec.fps << " rec_p95_ms=" << rec.p95_ms << " "
                      << "anon_fps=" << ano.fps << " anon_p95_ms=" << ano.p95_ms << " "
                      << "enc_fps=" << enc.fps << " enc_p95_ms=" << enc.p95_ms << " "
                      << "e2e_p95_ms=" << e2e.p95_ms
                      << "\n";

            for (const auto& [name, q] : queues) {
                if (q.capacity == 0) continue;
                if ((100 * q.size) / q.capacity >= 80) {
                    std::cerr << "[Metrics] queue_high_watermark " << name
                              << " size=" << q.size
                              << " cap=" << q.capacity
                              << " dropped=" << q.dropped
                              << "\n";
                }
            }
        }
    }
}
