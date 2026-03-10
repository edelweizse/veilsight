#include <common/config.hpp>
#include <common/replicate.hpp>
#include <encode/mjpeg_server.hpp>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <csignal>
#include <exception>

#include <pipeline/runtime.hpp>

static std::atomic<bool> g_running(true);
static void handle_sigint(int) { g_running = false; }

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    std::string cfg_path = "../../../configs/dual_example.yaml";
    if (argc >= 2) cfg_path = argv[1];
    else std::cerr << "Using default config: " << cfg_path << "\n";

    ss::AppConfig cfg;
    try {
        cfg = ss::load_config_yaml(cfg_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    std::vector<ss::IngestConfig> streams = ss::expand_replicas(cfg.streams);
    if (streams.empty()) {
        std::cerr << "No streams configured";
        return 1;
    }

    ss::MJPEGServer server(cfg.server.url, cfg.server.port);
    if (!server.start()) {
        std::cerr << "Server failed to start\n";
        return 1;
    }

    for (const auto& s : streams) {
        server.register_stream(s.id + "/ui");
    }

    ss::PipelineRuntime::Options opt;
    opt.jpeg_quality = 75;
    opt.detector = cfg.modules.detector;
    opt.tracker = cfg.modules.tracker;
    opt.recognizer = cfg.modules.recognizer;
    opt.metrics = cfg.metrics;

    opt.anonymizer_method = "blur"; // "pixelate" | "blur"
    opt.anonymizer_pixelation_divisor = 15;
    opt.anonymizer_blur_kernel = 75;

    ss::PipelineRuntime rt(server, streams, opt);
    if (!rt.start()) {
        std::cerr << "Pipeline failed to start\n";
        server.stop();
        return 1;
    }

    while (g_running) std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cerr << "Shutting down...\n";

    rt.stop();
    server.stop();
    return 0;
}
