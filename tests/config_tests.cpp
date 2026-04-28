#include <common/config.hpp>
#include <common/replicate.hpp>
#include <streaming/webrtc_publisher.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    int g_failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++g_failures;
            std::cerr << "[FAIL] " << message << "\n";
        }
    }

    std::string write_yaml_file(const std::string& prefix, const std::string& body) {
        namespace fs = std::filesystem;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const fs::path path = fs::temp_directory_path() /
                              (prefix + "_" + std::to_string(stamp) + ".yaml");

        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("failed to open temp config file: " + path.string());
        }
        out << body;
        out.close();
        return path.string();
    }

    bool load_throws(const std::string& yaml) {
        const std::string path = write_yaml_file("veilsight_cfg", yaml);
        try {
            (void)veilsight::load_config_yaml(path);
            std::filesystem::remove(path);
            return false;
        } catch (...) {
            std::filesystem::remove(path);
            return true;
        }
    }

    void test_expand_replicas_fills_missing_ids() {
        veilsight::IngestConfig in_cfg;
        in_cfg.id = "cam0";
        in_cfg.type = "webcam";
        in_cfg.replicate.count = 3;
        in_cfg.replicate.ids = {"custom_0"};

        const std::vector<veilsight::IngestConfig> input = {in_cfg};
        const auto expanded = veilsight::expand_replicas(input);

        check(expanded.size() == 3, "expand_replicas should output replicate.count entries");
        check(expanded[0].id == "custom_0", "expand_replicas should preserve provided ids");
        check(expanded[1].id == "cam0_1", "expand_replicas should synthesize missing id #1");
        check(expanded[2].id == "cam0_2", "expand_replicas should synthesize missing id #2");
    }

    void test_config_rejects_legacy_output() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    output:\n"
            "      width: 1280\n"
            "      height: 720\n";

        check(load_throws(yaml), "load_config_yaml should reject legacy stream.output schema");
    }

    void test_config_requires_global_outputs_fps() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n";

        check(load_throws(yaml), "load_config_yaml should require outputs.fps > 0");
    }

    void test_global_outputs_fps_overrides_profile_fps() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      fps: 12\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "          fps: 5\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n"
            "          fps: 30\n";

        const std::string path = write_yaml_file("veilsight_cfg_ok", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.streams.size() == 1, "valid config should load exactly one stream");
        const auto& profiles = cfg.streams[0].outputs.profiles;
        check(cfg.streams[0].outputs.fps == 12, "outputs.fps should be stored");
        check(profiles.at("inference").fps == 12, "inference fps should be synchronized to outputs.fps");
        check(profiles.at("ui").fps == 12, "ui fps should be synchronized to outputs.fps");
    }

    std::string base_streaming_yaml(const std::string& streaming) {
        return
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streaming:\n" + streaming +
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      fps: 12\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n";
    }

    void test_streaming_webrtc_defaults_and_parsing() {
        const std::string yaml = base_streaming_yaml(
            "  primary: \"webrtc\"\n"
            "  fallback: \"mjpeg\"\n"
            "  codec: \"h264\"\n"
            "  encoder: \"auto\"\n"
            "  bitrate_kbps: 2500\n"
            "  keyframe_interval_frames: 30\n"
            "  webrtc:\n"
            "    enabled: true\n"
            "    max_peers_per_stream: 3\n"
            "    ice_gathering_timeout_ms: 1500\n"
            "    session_idle_timeout_s: 45\n"
            "    stun_servers:\n"
            "      - \"stun:stun.l.google.com:19302\"\n"
            "    cors_allowed_origins:\n"
            "      - \"http://localhost:8000\"\n");

        const std::string path = write_yaml_file("veilsight_streaming_ok", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.streaming.primary == "webrtc", "streaming.primary should parse");
        check(cfg.streaming.webrtc.max_peers_per_stream == 3, "webrtc peer limit should parse");
        check(cfg.streaming.webrtc.ice_gathering_timeout_ms == 1500, "ICE timeout should parse");
        check(cfg.streaming.webrtc.stun_servers.size() == 1, "STUN servers should parse");
    }

    void test_streaming_validation_rejects_invalid_values() {
        check(load_throws(base_streaming_yaml("  bitrate_kbps: 0\n")),
              "streaming.bitrate_kbps must reject non-positive values");
        check(load_throws(base_streaming_yaml("  codec: \"vp8\"\n")),
              "streaming.codec must reject non-h264 values");
        check(load_throws(base_streaming_yaml("  webrtc:\n    max_peers_per_stream: 0\n")),
              "streaming.webrtc.max_peers_per_stream must reject zero");
        check(load_throws(base_streaming_yaml("  webrtc:\n    ice_gathering_timeout_ms: 50\n")),
              "streaming.webrtc.ice_gathering_timeout_ms must reject low timeout");
    }

    void test_h264_encoder_selector() {
        veilsight::StreamingConfig cfg;
        cfg.encoder = "x264enc";
        veilsight::H264EncoderSelector exact([](const std::string& name) {
            return name == "webrtcbin" || name == "x264enc";
        });
        auto selected = exact.select(cfg);
        check(selected.available && selected.encoder == "x264enc",
              "exact configured encoder should be selected when available");

        cfg.encoder = "auto";
        veilsight::H264EncoderSelector automatic([](const std::string& name) {
            return name == "webrtcbin" || name == "openh264enc";
        });
        selected = automatic.select(cfg);
        check(selected.available && selected.encoder == "openh264enc",
              "auto selector should pick the first available fake encoder");

        cfg.encoder = "missingenc";
        selected = exact.select(cfg);
        check(!selected.available && selected.error.find("missingenc") != std::string::npos,
              "missing exact encoder should return a clear error");
    }
}

int main() {
    test_expand_replicas_fills_missing_ids();
    test_config_rejects_legacy_output();
    test_config_requires_global_outputs_fps();
    test_global_outputs_fps_overrides_profile_fps();
    test_streaming_webrtc_defaults_and_parsing();
    test_streaming_validation_rejects_invalid_values();
    test_h264_encoder_selector();

    if (g_failures != 0) {
        std::cerr << "[FAIL] total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "[OK] all config tests passed\n";
    return 0;
}
