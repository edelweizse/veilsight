#pragma once

#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "runner_manager.hpp"
#include "telemetry_hub.hpp"
#include "veilsight/runner/v1/runner.grpc.pb.h"

namespace veilsight {
    class RunnerGrpcServer {
    public:
        RunnerGrpcServer(RunnerManager& manager, TelemetryHub& telemetry);
        ~RunnerGrpcServer();

        bool start(const RunnerGrpcConfig& config);
        void stop();
        void wait();

    private:
        class ControlService;
        class TelemetryService;

        RunnerManager& manager_;
        TelemetryHub& telemetry_;
        std::unique_ptr<ControlService> control_service_;
        std::unique_ptr<TelemetryService> telemetry_service_;
        std::unique_ptr<grpc::Server> server_;
    };
}
