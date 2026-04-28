#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <pipeline/publishers.hpp>

#include "veilsight/runner/v1/runner.pb.h"

namespace veilsight {
    class TelemetryHub final : public ITelemetryPublisher {
    public:
        using TelemetryEvent = runner::v1::TelemetryEvent;
        using MetricsSnapshot = runner::v1::MetricsSnapshot;

        explicit TelemetryHub(std::string runner_id, size_t subscriber_capacity = 512);
        ~TelemetryHub() override;

        class Subscription {
        public:
            Subscription(TelemetryHub& hub, uint64_t id);
            Subscription(const Subscription&) = delete;
            Subscription& operator=(const Subscription&) = delete;
            Subscription(Subscription&& other) noexcept;
            Subscription& operator=(Subscription&& other) noexcept;
            ~Subscription();

            bool pop_for(TelemetryEvent& out, std::chrono::milliseconds timeout);
            void close();

        private:
            TelemetryHub* hub_ = nullptr;
            uint64_t id_ = 0;
        };

        Subscription subscribe();
        MetricsSnapshot latest_metrics() const;
        void set_streams(std::vector<std::string> stream_ids);

        void publish_frame_analytics(const FrameCtx& frame,
                                     const std::vector<Box>& tracks) override;
        void publish_metrics_snapshot(const RuntimeMetrics::Snapshot& snapshot,
                                      const std::map<std::string, QueueSnapshot>& queues) override;
        void publish_status(std::string state, std::string message) override;
        void publish_log(std::string level, std::string message) override;

    private:
        struct Subscriber {
            std::mutex mutex;
            std::condition_variable cv;
            std::deque<TelemetryEvent> queue;
            bool closed = false;
        };

        static uint64_t now_epoch_ms_();
        void publish_(TelemetryEvent event, bool coalesce_metrics);
        void unsubscribe_(uint64_t id);
        std::shared_ptr<Subscriber> subscriber_(uint64_t id) const;

        std::string runner_id_;
        size_t subscriber_capacity_;

        mutable std::mutex mutex_;
        uint64_t next_subscriber_id_ = 1;
        std::map<uint64_t, std::shared_ptr<Subscriber>> subscribers_;
        MetricsSnapshot latest_metrics_;
        std::vector<std::string> stream_ids_;
    };
}
