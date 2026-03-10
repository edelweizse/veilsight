#include <pipeline/metrics.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace ss {
    namespace {
        uint64_t now_epoch_ms() {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count());
        }

        uint64_t now_steady_ns() {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
        }

        struct Histogram {
            static constexpr std::array<uint64_t, 17> kBoundsUs = {
                50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000,
                50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000};
            std::array<uint64_t, kBoundsUs.size() + 1> bins{};

            void observe(uint64_t duration_ns) {
                const uint64_t us = duration_ns / 1000;
                const auto it = std::lower_bound(kBoundsUs.begin(), kBoundsUs.end(), us);
                const size_t idx = static_cast<size_t>(std::distance(kBoundsUs.begin(), it));
                ++bins[idx];
            }

            double percentile_ms(double p, uint64_t count, uint64_t max_ns) const {
                if (count == 0) return 0.0;
                const uint64_t target = static_cast<uint64_t>(std::ceil(p * static_cast<double>(count)));
                uint64_t seen = 0;
                for (size_t i = 0; i < bins.size(); ++i) {
                    seen += bins[i];
                    if (seen >= target) {
                        if (i < kBoundsUs.size()) {
                            return static_cast<double>(kBoundsUs[i]) / 1000.0;
                        }
                        return static_cast<double>(max_ns) / 1e6;
                    }
                }
                return static_cast<double>(max_ns) / 1e6;
            }
        };

        struct StageAccumulator {
            uint64_t count = 0;
            uint64_t errors = 0;
            long double sum_ns = 0.0;
            uint64_t min_ns = std::numeric_limits<uint64_t>::max();
            uint64_t max_ns = 0;
            Histogram histogram;

            void observe(uint64_t duration_ns, bool ok) {
                ++count;
                if (!ok) ++errors;
                sum_ns += static_cast<long double>(duration_ns);
                min_ns = std::min(min_ns, duration_ns);
                max_ns = std::max(max_ns, duration_ns);
                histogram.observe(duration_ns);
            }

            StageSnapshot snapshot(double uptime_s) const {
                StageSnapshot out;
                out.count = count;
                out.errors = errors;
                out.fps = (uptime_s > 0.0) ? static_cast<double>(count) / uptime_s : 0.0;
                if (count == 0) return out;

                out.avg_ms = static_cast<double>(sum_ns / static_cast<long double>(count)) / 1e6;
                out.min_ms = static_cast<double>(min_ns) / 1e6;
                out.max_ms = static_cast<double>(max_ns) / 1e6;
                out.p50_ms = histogram.percentile_ms(0.50, count, max_ns);
                out.p95_ms = histogram.percentile_ms(0.95, count, max_ns);
                out.p99_ms = histogram.percentile_ms(0.99, count, max_ns);
                return out;
            }
        };

        std::string json_escape(const std::string& s) {
            std::string out;
            out.reserve(s.size() + 8);
            for (char c : s) {
                switch (c) {
                    case '\\': out += "\\\\"; break;
                    case '\"': out += "\\\""; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    case '\t': out += "\\t"; break;
                    default: out.push_back(c); break;
                }
            }
            return out;
        }

        void append_stage_snapshot_json(std::ostringstream& oss, const StageSnapshot& s) {
            oss << "{"
                << "\"count\":" << s.count << ","
                << "\"errors\":" << s.errors << ","
                << "\"fps\":" << s.fps << ","
                << "\"avg_ms\":" << s.avg_ms << ","
                << "\"p50_ms\":" << s.p50_ms << ","
                << "\"p95_ms\":" << s.p95_ms << ","
                << "\"p99_ms\":" << s.p99_ms << ","
                << "\"min_ms\":" << s.min_ms << ","
                << "\"max_ms\":" << s.max_ms
                << "}";
        }
    } // namespace

    const char* runtime_stage_name(RuntimeStage stage) {
        switch (stage) {
            case RuntimeStage::Ingest: return "ingest";
            case RuntimeStage::Detector: return "detector";
            case RuntimeStage::Tracker: return "tracker";
            case RuntimeStage::Recognizer: return "recognizer";
            case RuntimeStage::Anonymizer: return "anonymizer";
            case RuntimeStage::Encoder: return "encoder";
            case RuntimeStage::EndToEnd: return "end_to_end";
        }
        return "unknown";
    }

    const std::vector<RuntimeStage>& runtime_stage_order() {
        static const std::vector<RuntimeStage> kOrder = {
            RuntimeStage::Ingest,
            RuntimeStage::Detector,
            RuntimeStage::Tracker,
            RuntimeStage::Recognizer,
            RuntimeStage::Anonymizer,
            RuntimeStage::Encoder,
            RuntimeStage::EndToEnd
        };
        return kOrder;
    }

    struct RuntimeMetrics::Impl {
        uint64_t started_ns = now_steady_ns();
        mutable std::mutex mutex;
        std::map<RuntimeStage, StageAccumulator> global;
        std::map<std::string, std::map<RuntimeStage, StageAccumulator>> streams;
    };

    RuntimeMetrics::RuntimeMetrics()
        : impl_(new Impl()) {}

    RuntimeMetrics::~RuntimeMetrics() {
        delete impl_;
        impl_ = nullptr;
    }

    void RuntimeMetrics::observe_global(RuntimeStage stage, uint64_t duration_ns, bool ok) {
        std::lock_guard lk(impl_->mutex);
        impl_->global[stage].observe(duration_ns, ok);
    }

    void RuntimeMetrics::observe_stream(const std::string& stream_id,
                                        RuntimeStage stage,
                                        uint64_t duration_ns,
                                        bool ok) {
        std::lock_guard lk(impl_->mutex);
        impl_->streams[stream_id][stage].observe(duration_ns, ok);
    }

    RuntimeMetrics::Snapshot RuntimeMetrics::snapshot() const {
        const uint64_t now_ns = now_steady_ns();
        const double uptime_s = static_cast<double>(now_ns - impl_->started_ns) / 1e9;

        RuntimeMetrics::Snapshot out;
        out.timestamp_ms = now_epoch_ms();
        out.uptime_s = uptime_s;

        std::lock_guard lk(impl_->mutex);

        for (const auto& [stage, acc] : impl_->global) {
            out.global[stage] = acc.snapshot(uptime_s);
        }
        for (const auto& [stream, stage_map] : impl_->streams) {
            auto& stream_out = out.streams[stream];
            for (const auto& [stage, acc] : stage_map) {
                stream_out[stage] = acc.snapshot(uptime_s);
            }
        }
        return out;
    }

    std::string metrics_snapshot_to_json(const RuntimeMetrics::Snapshot& snapshot,
                                         const std::map<std::string, QueueSnapshot>& queues) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(3);

        oss << "{"
            << "\"timestamp_ms\":" << snapshot.timestamp_ms << ","
            << "\"uptime_s\":" << snapshot.uptime_s << ",";

        oss << "\"global\":{";
        bool first_stage = true;
        for (RuntimeStage stage : runtime_stage_order()) {
            auto it = snapshot.global.find(stage);
            if (it == snapshot.global.end()) continue;
            if (!first_stage) oss << ",";
            first_stage = false;
            oss << "\"" << runtime_stage_name(stage) << "\":";
            append_stage_snapshot_json(oss, it->second);
        }
        oss << "},";

        oss << "\"streams\":{";
        bool first_stream = true;
        for (const auto& [stream_id, stage_map] : snapshot.streams) {
            if (!first_stream) oss << ",";
            first_stream = false;
            oss << "\"" << json_escape(stream_id) << "\":{";
            bool inner_first = true;
            for (RuntimeStage stage : runtime_stage_order()) {
                auto it = stage_map.find(stage);
                if (it == stage_map.end()) continue;
                if (!inner_first) oss << ",";
                inner_first = false;
                oss << "\"" << runtime_stage_name(stage) << "\":";
                append_stage_snapshot_json(oss, it->second);
            }
            oss << "}";
        }
        oss << "},";

        oss << "\"queues\":{";
        bool first_queue = true;
        for (const auto& [name, q] : queues) {
            if (!first_queue) oss << ",";
            first_queue = false;
            oss << "\"" << json_escape(name) << "\":"
                << "{"
                << "\"size\":" << q.size << ","
                << "\"capacity\":" << q.capacity << ","
                << "\"dropped\":" << q.dropped
                << "}";
        }
        oss << "}";

        oss << "}";
        return oss.str();
    }
}
