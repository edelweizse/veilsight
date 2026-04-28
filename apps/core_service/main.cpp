#include <common/config.hpp>
#include <common/replicate.hpp>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <csignal>
#include <exception>

#include "grpc_runner_server.hpp"
#include "runner_manager.hpp"
#include "telemetry_hub.hpp"

static std::atomic<bool> g_running(true);
static void handle_sigint(int) { g_running = false; }

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    std::string cfg_path = "../../../configs/dual_example.yaml";
    bool autostart = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-autostart") {
            autostart = false;
        } else {
            cfg_path = arg;
        }
    }
    if (argc < 2) {
        std::cerr << "Using default config: " << cfg_path << "\n";
    }

    veilsight::AppConfig cfg;
    try {
        cfg = veilsight::load_config_yaml(cfg_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    veilsight::TelemetryHub telemetry(cfg.runner.id);
    veilsight::RunnerManager runner(cfg_path, cfg, telemetry);
    veilsight::RunnerGrpcServer grpc_server(runner, telemetry);
    if (!grpc_server.start(cfg.runner.grpc)) {
        std::cerr << "Runner gRPC failed to start\n";
        return 1;
    }

    if (autostart) {
        std::string start_error;
        if (!runner.start(&start_error)) {
            std::cerr << "Pipeline failed to start; runner control service remains available\n";
            if (!start_error.empty()) std::cerr << start_error << "\n";
        }
    }

    while (g_running) std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cerr << "Shutting down...\n";

    runner.stop();
    grpc_server.stop();
    return 0;
}
