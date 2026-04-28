#include "telemetry_hub.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace veilsight {
    namespace {
        void fill_stage(runner::v1::StageMetrics* out,
                        const std::string& name,
                        const StageSnapshot& in) {
            out->set_stage(name);
            out->set_count(in.count);
            out->set_errors(in.errors);
            out->set_fps(in.fps);
            out->set_avg_ms(in.avg_ms);
            out->set_p50_ms(in.p50_ms);
            out->set_p95_ms(in.p95_ms);
            out->set_p99_ms(in.p99_ms);
        }
    }

    TelemetryHub::TelemetryHub(std::string runner_id, size_t subscriber_capacity)
        : runner_id_(std::move(runner_id)),
          subscriber_capacity_(subscriber_capacity) {}

    TelemetryHub::~TelemetryHub() {
        std::vector<std::shared_ptr<Subscriber>> subscribers;
        {
            std::lock_guard lk(mutex_);
            for (auto& [_, sub] : subscribers_) subscribers.push_back(sub);
            subscribers_.clear();
        }
        for (auto& sub : subscribers) {
            std::lock_guard lk(sub->mutex);
            sub->closed = true;
            sub->cv.notify_all();
        }
    }

    TelemetryHub::Subscription::Subscription(TelemetryHub& hub, uint64_t id)
        : hub_(&hub), id_(id) {}

    TelemetryHub::Subscription::Subscription(Subscription&& other) noexcept
        : hub_(other.hub_), id_(other.id_) {
        other.hub_ = nullptr;
        other.id_ = 0;
    }

    TelemetryHub::Subscription& TelemetryHub::Subscription::operator=(Subscription&& other) noexcept {
        if (this == &other) return *this;
        close();
        hub_ = other.hub_;
        id_ = other.id_;
        other.hub_ = nullptr;
        other.id_ = 0;
        return *this;
    }

    TelemetryHub::Subscription::~Subscription() {
        close();
    }

    bool TelemetryHub::Subscription::pop_for(TelemetryEvent& out, std::chrono::milliseconds timeout) {
        if (!hub_) return false;
        auto sub = hub_->subscriber_(id_);
        if (!sub) return false;

        std::unique_lock lk(sub->mutex);
        if (!sub->cv.wait_for(lk, timeout, [&] { return sub->closed || !sub->queue.empty(); })) {
            return false;
        }
        if (sub->closed || sub->queue.empty()) return false;
        out = std::move(sub->queue.front());
        sub->queue.pop_front();
        return true;
    }

    void TelemetryHub::Subscription::close() {
        if (!hub_) return;
        hub_->unsubscribe_(id_);
        hub_ = nullptr;
        id_ = 0;
    }

    TelemetryHub::Subscription TelemetryHub::subscribe() {
        std::lock_guard lk(mutex_);
        const uint64_t id = next_subscriber_id_++;
        subscribers_[id] = std::make_shared<Subscriber>();
        return Subscription(*this, id);
    }

    TelemetryHub::MetricsSnapshot TelemetryHub::latest_metrics() const {
        std::lock_guard lk(mutex_);
        return latest_metrics_;
    }

    void TelemetryHub::set_streams(std::vector<std::string> stream_ids) {
        std::lock_guard lk(mutex_);
        stream_ids_ = std::move(stream_ids);
    }

    void TelemetryHub::publish_frame_analytics(const FrameCtx& frame,
                                               const std::vector<Box>& tracks) {
        TelemetryEvent event;
        auto* out = event.mutable_frame();
        out->mutable_stream()->set_stream_id(frame.stream_id);
        out->mutable_stream()->set_profile("ui");
        out->set_frame_id(frame.frame_id);
        out->set_pts_ns(frame.pts_ns);
        out->set_width(frame.ui_w);
        out->set_height(frame.ui_h);
        for (const auto& box : tracks) {
            auto* t = out->add_tracks();
            t->set_id(box.id);
            t->set_x(box.x);
            t->set_y(box.y);
            t->set_w(box.w);
            t->set_h(box.h);
            t->set_score(box.score);
            t->set_occluded(box.occluded);
        }
        publish_(std::move(event), false);
    }

    void TelemetryHub::publish_metrics_snapshot(const RuntimeMetrics::Snapshot& snapshot,
                                                const std::map<std::string, QueueSnapshot>& queues) {
        TelemetryEvent event;
        auto* out = event.mutable_metrics();
        out->set_timestamp_ms(snapshot.timestamp_ms);
        out->set_uptime_s(snapshot.uptime_s);

        for (RuntimeStage stage : runtime_stage_order()) {
            auto it = snapshot.global.find(stage);
            if (it == snapshot.global.end()) continue;
            fill_stage(out->add_global(), runtime_stage_name(stage), it->second);
        }

        for (const auto& [stream_id, stages] : snapshot.streams) {
            auto* stream = out->add_streams();
            stream->mutable_stream()->set_stream_id(stream_id);
            stream->mutable_stream()->set_profile("ui");
            for (RuntimeStage stage : runtime_stage_order()) {
                auto it = stages.find(stage);
                if (it == stages.end()) continue;
                fill_stage(stream->add_stages(), runtime_stage_name(stage), it->second);
            }
        }

        for (const auto& [name, queue] : queues) {
            auto* q = out->add_queues();
            q->set_name(name);
            q->set_size(queue.size);
            q->set_capacity(queue.capacity);
            q->set_dropped(queue.dropped);
        }

        {
            std::lock_guard lk(mutex_);
            latest_metrics_ = *out;
        }
        publish_(std::move(event), true);
    }

    void TelemetryHub::publish_status(std::string state, std::string message) {
        TelemetryEvent event;
        auto* status = event.mutable_status();
        status->set_runner_id(runner_id_);
        status->set_state(std::move(state));
        status->set_message(std::move(message));
        status->set_timestamp_ms(now_epoch_ms_());
        publish_(std::move(event), false);
    }

    void TelemetryHub::publish_log(std::string level, std::string message) {
        TelemetryEvent event;
        auto* log = event.mutable_log();
        log->set_timestamp_ms(now_epoch_ms_());
        log->set_level(std::move(level));
        log->set_message(std::move(message));
        publish_(std::move(event), false);
    }

    uint64_t TelemetryHub::now_epoch_ms_() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
    }

    void TelemetryHub::publish_(TelemetryEvent event, bool coalesce_metrics) {
        std::vector<std::shared_ptr<Subscriber>> subscribers;
        {
            std::lock_guard lk(mutex_);
            for (auto& [_, sub] : subscribers_) subscribers.push_back(sub);
        }

        for (auto& sub : subscribers) {
            {
                std::lock_guard lk(sub->mutex);
                if (sub->closed) continue;
                if (coalesce_metrics) {
                    sub->queue.erase(std::remove_if(sub->queue.begin(),
                                                    sub->queue.end(),
                                                    [](const TelemetryEvent& queued) {
                                                        return queued.has_metrics();
                                                    }),
                                     sub->queue.end());
                }
                while (sub->queue.size() >= subscriber_capacity_) {
                    sub->queue.pop_front();
                }
                sub->queue.push_back(event);
            }
            sub->cv.notify_one();
        }
    }

    void TelemetryHub::unsubscribe_(uint64_t id) {
        std::shared_ptr<Subscriber> sub;
        {
            std::lock_guard lk(mutex_);
            auto it = subscribers_.find(id);
            if (it == subscribers_.end()) return;
            sub = it->second;
            subscribers_.erase(it);
        }
        {
            std::lock_guard lk(sub->mutex);
            sub->closed = true;
        }
        sub->cv.notify_all();
    }

    std::shared_ptr<TelemetryHub::Subscriber> TelemetryHub::subscriber_(uint64_t id) const {
        std::lock_guard lk(mutex_);
        auto it = subscribers_.find(id);
        return it == subscribers_.end() ? nullptr : it->second;
    }
}
