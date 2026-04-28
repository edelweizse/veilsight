#include "grpc_runner_server.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <set>
#include <thread>

namespace veilsight {
    namespace {
        uint64_t now_epoch_ms() {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count());
        }

        std::string grpc_bind_address(const std::string& configured) {
            constexpr std::string_view unix_prefix = "unix://";
            if (configured.rfind(unix_prefix, 0) == 0) {
                return "unix:" + configured.substr(unix_prefix.size());
            }
            if (configured.rfind("unix:", 0) == 0) return configured;
            return configured;
        }

        void unlink_unix_socket_if_needed(const std::string& bind_address) {
            constexpr std::string_view prefix = "unix:";
            if (bind_address.rfind(prefix, 0) != 0) return;
            const std::string path = bind_address.substr(prefix.size());
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    class RunnerGrpcServer::ControlService final : public runner::v1::PipelineControlService::Service {
    public:
        explicit ControlService(RunnerManager& manager) : manager_(manager) {}

        grpc::Status Health(grpc::ServerContext*,
                            const runner::v1::HealthRequest*,
                            runner::v1::HealthResponse* response) override {
            response->set_ok(true);
            response->set_runner_id(manager_.status().runner_id());
            response->set_version("dev");
            response->set_timestamp_ms(now_epoch_ms());
            return grpc::Status::OK;
        }

        grpc::Status GetStatus(grpc::ServerContext*,
                               const runner::v1::GetStatusRequest*,
                               runner::v1::PipelineStatus* response) override {
            *response = manager_.status();
            return grpc::Status::OK;
        }

        grpc::Status GetStreams(grpc::ServerContext*,
                                const runner::v1::GetStreamsRequest*,
                                runner::v1::StreamList* response) override {
            *response = manager_.streams();
            return grpc::Status::OK;
        }

        grpc::Status ValidateConfig(grpc::ServerContext*,
                                    const runner::v1::ValidateConfigRequest* request,
                                    runner::v1::ValidateConfigResponse* response) override {
            std::vector<std::string> errors;
            const bool valid = manager_.validate_config_yaml(request->config_yaml(), &errors);
            response->set_valid(valid);
            for (const auto& error : errors) response->add_errors(error);
            return grpc::Status::OK;
        }

        grpc::Status Start(grpc::ServerContext*,
                           const runner::v1::StartRequest* request,
                           runner::v1::PipelineCommandResponse* response) override {
            std::string error;
            bool ok = true;
            if (!request->config_yaml().empty()) {
                ok = manager_.reload_config_yaml(request->config_yaml(), &error);
            } else {
                ok = manager_.start(&error);
            }
            response->set_accepted(ok);
            response->set_message(ok ? "started" : error);
            *response->mutable_status() = manager_.status(response->message());
            return grpc::Status::OK;
        }

        grpc::Status Stop(grpc::ServerContext*,
                          const runner::v1::StopRequest*,
                          runner::v1::PipelineCommandResponse* response) override {
            manager_.stop();
            response->set_accepted(true);
            response->set_message("stopped");
            *response->mutable_status() = manager_.status("stopped");
            return grpc::Status::OK;
        }

        grpc::Status Reload(grpc::ServerContext*,
                            const runner::v1::ReloadRequest* request,
                            runner::v1::PipelineCommandResponse* response) override {
            std::string error;
            const bool ok = manager_.reload_config_yaml(request->config_yaml(), &error);
            response->set_accepted(ok);
            response->set_message(ok ? "reloaded" : error);
            *response->mutable_status() = manager_.status(response->message());
            return grpc::Status::OK;
        }

    private:
        RunnerManager& manager_;
    };

    class RunnerGrpcServer::TelemetryService final : public runner::v1::RunnerTelemetryService::Service {
    public:
        explicit TelemetryService(TelemetryHub& telemetry) : telemetry_(telemetry) {}

        grpc::Status WatchTelemetry(grpc::ServerContext* context,
                                    const runner::v1::WatchTelemetryRequest* request,
                                    grpc::ServerWriter<runner::v1::TelemetryEvent>* writer) override {
            auto sub = telemetry_.subscribe();
            const std::set<std::string> stream_filter(request->stream_ids().begin(), request->stream_ids().end());
            while (!context->IsCancelled()) {
                runner::v1::TelemetryEvent event;
                if (!sub.pop_for(event, std::chrono::milliseconds(250))) continue;
                if (event.has_frame()) {
                    if (!request->include_frame_analytics()) continue;
                    if (!stream_filter.empty() &&
                        stream_filter.find(event.frame().stream().stream_id()) == stream_filter.end()) {
                        continue;
                    }
                } else if (event.has_metrics()) {
                    if (!request->include_metrics()) continue;
                } else if (event.has_log()) {
                    if (!request->include_logs()) continue;
                }
                if (!writer->Write(event)) break;
            }
            return grpc::Status::OK;
        }

        grpc::Status GetMetricsSnapshot(grpc::ServerContext*,
                                        const runner::v1::GetMetricsSnapshotRequest*,
                                        runner::v1::MetricsSnapshot* response) override {
            *response = telemetry_.latest_metrics();
            return grpc::Status::OK;
        }

    private:
        TelemetryHub& telemetry_;
    };

    RunnerGrpcServer::RunnerGrpcServer(RunnerManager& manager, TelemetryHub& telemetry)
        : manager_(manager),
          telemetry_(telemetry),
          control_service_(std::make_unique<ControlService>(manager_)),
          telemetry_service_(std::make_unique<TelemetryService>(telemetry_)) {}

    RunnerGrpcServer::~RunnerGrpcServer() {
        stop();
    }

    bool RunnerGrpcServer::start(const RunnerGrpcConfig& config) {
        grpc::ServerBuilder builder;
        const auto listen = grpc_bind_address(config.listen);
        unlink_unix_socket_if_needed(listen);

        int selected_port = 0;
        builder.AddListeningPort(listen, grpc::InsecureServerCredentials(), &selected_port);
        if (!config.fallback_tcp.empty()) {
            builder.AddListeningPort(config.fallback_tcp, grpc::InsecureServerCredentials());
        }
        builder.RegisterService(control_service_.get());
        builder.RegisterService(telemetry_service_.get());

        server_ = builder.BuildAndStart();
        if (!server_) {
            std::cerr << "[Runner gRPC] failed to start\n";
            return false;
        }
        std::cerr << "[Runner gRPC] listening on " << config.listen;
        if (!config.fallback_tcp.empty()) std::cerr << " and " << config.fallback_tcp;
        std::cerr << "\n";
        return true;
    }

    void RunnerGrpcServer::stop() {
        if (!server_) return;
        server_->Shutdown();
        server_.reset();
    }

    void RunnerGrpcServer::wait() {
        if (server_) server_->Wait();
    }
}
