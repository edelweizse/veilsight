#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ss {
    enum class RuntimeStage {
        Ingest,
        Detector,
        Tracker,
        Recognizer,
        Anonymizer,
        Encoder,
        EndToEnd
    };

    const char* runtime_stage_name(RuntimeStage stage);
    const std::vector<RuntimeStage>& runtime_stage_order();

    struct StageSnapshot {
        uint64_t count = 0;
        uint64_t errors = 0;
        double fps = 0.0;
        double avg_ms = 0.0;
        double p50_ms = 0.0;
        double p95_ms = 0.0;
        double p99_ms = 0.0;
        double min_ms = 0.0;
        double max_ms = 0.0;
    };

    struct QueueSnapshot {
        size_t size = 0;
        size_t capacity = 0;
        uint64_t dropped = 0;
    };

    class RuntimeMetrics {
    public:
        struct Snapshot {
            uint64_t timestamp_ms = 0;
            double uptime_s = 0.0;
            std::map<RuntimeStage, StageSnapshot> global;
            std::map<std::string, std::map<RuntimeStage, StageSnapshot>> streams;
        };

        RuntimeMetrics();
        ~RuntimeMetrics();

        RuntimeMetrics(const RuntimeMetrics&) = delete;
        RuntimeMetrics& operator=(const RuntimeMetrics&) = delete;
        RuntimeMetrics(RuntimeMetrics&&) = delete;
        RuntimeMetrics& operator=(RuntimeMetrics&&) = delete;

        void observe_global(RuntimeStage stage, uint64_t duration_ns, bool ok = true);
        void observe_stream(const std::string& stream_id, RuntimeStage stage, uint64_t duration_ns, bool ok = true);
        Snapshot snapshot() const;

    private:
        struct Impl;
        Impl* impl_ = nullptr;
    };

    std::string metrics_snapshot_to_json(const RuntimeMetrics::Snapshot& snapshot,
                                         const std::map<std::string, QueueSnapshot>& queues);
}
