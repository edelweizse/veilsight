#include <common/config.hpp>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace veilsight {
    static bool get_bool(
        const YAML::Node& n, const char* key, bool def) {
        return (n && n[key]) ? n[key].as<bool>() : def;
    }

    static int get_int(
        const YAML::Node& n, const char* key, int def) {
        return (n && n[key]) ? n[key].as<int>() : def;
    }

    static std::string get_str(
        const YAML::Node& n, const char* key, const std::string& def) {
        return (n && n[key]) ? n[key].as<std::string>() : def;
    }

    static float get_float(
        const YAML::Node& n, const char* key, float def) {
        return (n && n[key]) ? n[key].as<float>() : def;
    }

    static WebcamConfig parse_webcam_config(const YAML::Node& wc) {
        WebcamConfig c;
        if (!wc) return c;
        c.device = get_str(wc, "device", "/dev/video0");
        c.width = get_int(wc, "width", 1280);
        c.height = get_int(wc, "height", 720);
        //c.fps = get_int(wc, "fps", 30);
        c.mjpg = get_bool(wc, "mjpg", get_bool(wc, "mjpeg", true));
        return c;
    }

    static FileConfig parse_file_config(const YAML::Node& fc) {
        FileConfig c;
        if (!fc) return c;
        c.path = get_str(fc, "path", "/");
        c.fps = get_int(fc, "fps", 30);
        c.loop = get_bool(fc, "loop", false);
        return c;
    }

    static RTSPConfig parse_rtsp_config(const YAML::Node& rc) {
        RTSPConfig c;
        if (!rc) return c;
        c.url = get_str(rc, "url", "/");
        //c.fps = get_int(rc, "fps", 30);
        c.latency_ms = get_int(rc, "latency_ms", 1000);
        c.tcp = get_bool(rc, "tcp", true);
        return c;
    }

    static ReplicateConfig parse_replicate_config(const YAML::Node& r) {
        ReplicateConfig c;
        if (!r) return c;
        c.count = get_int(r, "count", 1);
        if (r["ids"]) c.ids = r["ids"].as<std::vector<std::string>>();
        if (c.count < 1) c.count = 1;
        return c;
    }

    static OutputConfig parse_output_config(const YAML::Node& o, const OutputConfig& def = {}) {
        OutputConfig c = def;
        if (!o) return c;
        c.width = get_int(o, "width", c.width);
        c.height = get_int(o, "height", c.height);
        c.fps = get_int(o, "fps", c.fps);
        c.format = get_str(o, "format", c.format);
        c.keep_aspect = get_bool(o, "keep_aspect", c.keep_aspect);
        c.interp = get_str(o, "interp", c.interp);
        c.jpeg_quality = get_int(o, "jpeg_quality", c.jpeg_quality);
        return c;
    }

    static OutputsConfig parse_outputs_config(const YAML::Node& o) {
        OutputsConfig out;
        if (!o) return out;
        out.fps = get_int(o, "fps", 0);
        auto profiles = o["profiles"];
        if (!profiles) return out;
        if (!profiles.IsMap()) {
            throw std::runtime_error ("[Config] profiles must be a map!");
        }

        for (auto it = profiles.begin(); it != profiles.end(); ++it) {
            const auto name = it->first.as<std::string>();
            const YAML::Node cfg = it->second;
            out.profiles[name] = parse_output_config(cfg, OutputConfig{});
        }

        if (out.fps > 0) {
            for (auto& kv : out.profiles) {
                kv.second.fps = out.fps;
            }
        }

        return out;
    }

    static YuNetModuleConfig parse_yunet_module_config(const YAML::Node& n) {
        YuNetModuleConfig cfg;
        if (!n) return cfg;

        cfg.param_path = get_str(n, "param_path", cfg.param_path);
        cfg.bin_path = get_str(n, "bin_path", cfg.bin_path);
        cfg.input_w = get_int(n, "input_w", cfg.input_w);
        cfg.input_h = get_int(n, "input_h", cfg.input_h);
        cfg.score_threshold = n["score_threshold"] ? n["score_threshold"].as<float>() : cfg.score_threshold;
        cfg.nms_threshold = n["nms_threshold"] ? n["nms_threshold"].as<float>() : cfg.nms_threshold;
        cfg.top_k = get_int(n, "top_k", cfg.top_k);
        cfg.ncnn_threads = get_int(n, "ncnn_threads", cfg.ncnn_threads);
        return cfg;
    }

    static SCRFDModuleConfig parse_scrfd_module_config(const YAML::Node& n) {
        SCRFDModuleConfig cfg;
        if (!n) return cfg;

        cfg.variant = get_str(n, "variant", cfg.variant);
        if (cfg.variant == "25g") {
            cfg.param_path = "models/detector/scrfd_25g/scrfd_25g.ncnn.param";
            cfg.bin_path = "models/detector/scrfd_25g/scrfd_25g.ncnn.bin";
        } else if (cfg.variant == "500m") {
            cfg.param_path = "models/detector/scrfd_500m/scrfd_500m.ncnn.param";
            cfg.bin_path = "models/detector/scrfd_500m/scrfd_500m.ncnn.bin";
        } else if (cfg.variant == "25g_landmarks") {
            cfg.param_path = "models/detector/scrfd_25g_landmarks/scrfd_25g_landmarks.ncnn.param";
            cfg.bin_path = "models/detector/scrfd_25g_landmarks/scrfd_25g_landmarks.ncnn.bin";
        } else if (cfg.variant == "500m_landmarks") {
            cfg.param_path = "models/detector/scrfd_500m_landmarks/scrfd_500m_landmarks.ncnn.param";
            cfg.bin_path = "models/detector/scrfd_500m_landmarks/scrfd_500m_landmarks.ncnn.bin";
        } else if (cfg.variant == "10g") {
            cfg.param_path = "models/detector/scrfd_10g/scrfd_10g.ncnn.param";
            cfg.bin_path = "models/detector/scrfd_10g/scrfd_10g.ncnn.bin";
        }
        cfg.param_path = get_str(n, "param_path", cfg.param_path);
        cfg.bin_path = get_str(n, "bin_path", cfg.bin_path);
        cfg.input_w = get_int(n, "input_w", cfg.input_w);
        cfg.input_h = get_int(n, "input_h", cfg.input_h);
        cfg.score_threshold = n["score_threshold"] ? n["score_threshold"].as<float>() : cfg.score_threshold;
        cfg.nms_threshold = n["nms_threshold"] ? n["nms_threshold"].as<float>() : cfg.nms_threshold;
        cfg.top_k = get_int(n, "top_k", cfg.top_k);
        cfg.ncnn_threads = get_int(n, "ncnn_threads", cfg.ncnn_threads);
        return cfg;
    }

    static YoloXModuleConfig parse_yolox_module_config(const YAML::Node& n) {
        YoloXModuleConfig cfg;
        if (!n) return cfg;

        cfg.variant = get_str(n, "variant", cfg.variant);
        if (cfg.variant == "nano") {
            cfg.model_path = "models/detector/bytetrack_nano";
            cfg.param_path = "models/detector/bytetrack_nano.ncnn.param";
            cfg.bin_path = "models/detector/bytetrack_nano.ncnn.bin";
        } else if (cfg.variant == "tiny") {
            cfg.model_path = "models/detector/bytetrack_tiny";
            cfg.param_path = "models/detector/bytetrack_tiny.ncnn.param";
            cfg.bin_path = "models/detector/bytetrack_tiny.ncnn.bin";
        }
        cfg.model_path = get_str(n, "model_path", cfg.model_path);
        cfg.param_path = get_str(n, "param_path", cfg.param_path);
        cfg.bin_path = get_str(n, "bin_path", cfg.bin_path);
        if (n["model_path"] && !n["param_path"] && !n["bin_path"]) {
            const std::string prefix = cfg.model_path;
            cfg.param_path = prefix.ends_with(".ncnn.param") ? prefix : prefix + ".ncnn.param";
            cfg.bin_path = prefix.ends_with(".ncnn.bin") ? prefix : prefix + ".ncnn.bin";
        }
        cfg.input_w = get_int(n, "input_w", cfg.input_w);
        cfg.input_h = get_int(n, "input_h", cfg.input_h);
        cfg.score_threshold = get_float(n, "score_threshold", cfg.score_threshold);
        cfg.nms_threshold = get_float(n, "nms_threshold", cfg.nms_threshold);
        cfg.top_k = get_int(n, "top_k", cfg.top_k);
        cfg.class_id = get_int(n, "class_id", get_int(n, "person_class_id", cfg.class_id));
        cfg.ncnn_threads = get_int(n, "ncnn_threads", cfg.ncnn_threads);
        cfg.letterbox = get_bool(n, "letterbox", cfg.letterbox);
        cfg.decoded_output = get_bool(n, "decoded_output", cfg.decoded_output);
        return cfg;
    }

    static SceneGridConfig parse_scene_grid_config(const YAML::Node& n, const SceneGridConfig& def = {}) {
        SceneGridConfig cfg = def;
        if (!n) return cfg;

        cfg.enabled = get_bool(n, "enabled", cfg.enabled);
        cfg.rows = get_int(n, "rows", cfg.rows);
        cfg.cols = get_int(n, "cols", cfg.cols);
        cfg.association_weight = get_float(n, "association_weight", cfg.association_weight);
        cfg.cell_distance_weight = get_float(n, "cell_distance_weight", cfg.cell_distance_weight);
        cfg.occupancy_weight = get_float(n, "occupancy_weight", cfg.occupancy_weight);
        cfg.transition_weight = get_float(n, "transition_weight", cfg.transition_weight);
        cfg.max_extra_cost = get_float(n, "max_extra_cost", cfg.max_extra_cost);
        cfg.occupancy_decay = get_float(n, "occupancy_decay", cfg.occupancy_decay);
        cfg.transition_decay = get_float(n, "transition_decay", cfg.transition_decay);
        cfg.warmup_frames = get_int(n, "warmup_frames", cfg.warmup_frames);
        return cfg;
    }

    static DetectorModuleConfig parse_detector_module_config(const YAML::Node& n) {
        DetectorModuleConfig cfg;
        if (!n) return cfg;

        cfg.type = get_str(n, "type", cfg.type);
        cfg.workers = get_int(n, "workers", cfg.workers);
        if (cfg.workers < 1) cfg.workers = 1;

        cfg.yunet = parse_yunet_module_config(n["yunet"]);
        cfg.scrfd = parse_scrfd_module_config(n["scrfd"]);
        cfg.yolox = parse_yolox_module_config(n["yolox"]);
        return cfg;
    }

    static TrackerModuleConfig parse_tracker_module_config(const YAML::Node& n) {
        TrackerModuleConfig cfg;
        if (!n) return cfg;

        cfg.type = get_str(n, "type", cfg.type);
        const YAML::Node demo = n["demo"];
        if (demo) {
            cfg.demo.high_thresh = get_float(demo, "high_thresh", cfg.demo.high_thresh);
            cfg.demo.low_thresh = get_float(demo, "low_thresh", cfg.demo.low_thresh);
            cfg.demo.match_iou_thresh = get_float(demo, "match_iou_thresh", cfg.demo.match_iou_thresh);
            cfg.demo.low_match_iou_thresh = get_float(demo, "low_match_iou_thresh", cfg.demo.low_match_iou_thresh);
            cfg.demo.min_hits = get_int(demo, "min_hits", cfg.demo.min_hits);
            cfg.demo.max_missed = get_int(demo, "max_missed", cfg.demo.max_missed);
        }

        const YAML::Node bytetrack = n["bytetrack"];
        if (bytetrack) {
            cfg.bytetrack.high_thresh = get_float(bytetrack, "high_thresh", cfg.bytetrack.high_thresh);
            cfg.bytetrack.low_thresh = get_float(bytetrack, "low_thresh", cfg.bytetrack.low_thresh);
            cfg.bytetrack.new_track_thresh = get_float(bytetrack, "new_track_thresh", cfg.bytetrack.new_track_thresh);
            cfg.bytetrack.match_iou_thresh = get_float(bytetrack, "match_iou_thresh", cfg.bytetrack.match_iou_thresh);
            cfg.bytetrack.low_match_iou_thresh = get_float(bytetrack, "low_match_iou_thresh", cfg.bytetrack.low_match_iou_thresh);
            cfg.bytetrack.unconfirmed_match_iou_thresh = get_float(bytetrack, "unconfirmed_match_iou_thresh", cfg.bytetrack.unconfirmed_match_iou_thresh);
            cfg.bytetrack.duplicate_iou_thresh = get_float(bytetrack, "duplicate_iou_thresh", cfg.bytetrack.duplicate_iou_thresh);
            cfg.bytetrack.track_buffer = get_int(bytetrack, "track_buffer", cfg.bytetrack.track_buffer);
            cfg.bytetrack.min_box_area = get_float(bytetrack, "min_box_area", cfg.bytetrack.min_box_area);
            cfg.bytetrack.fuse_score = get_bool(bytetrack, "fuse_score", cfg.bytetrack.fuse_score);
            cfg.bytetrack.scene_grid = parse_scene_grid_config(bytetrack["scene_grid"], cfg.bytetrack.scene_grid);
        }

        const YAML::Node ocsort = n["ocsort"];
        if (ocsort) {
            cfg.ocsort.det_thresh = get_float(ocsort, "det_thresh", cfg.ocsort.det_thresh);
            cfg.ocsort.low_det_thresh = get_float(ocsort, "low_det_thresh", cfg.ocsort.low_det_thresh);
            cfg.ocsort.iou_threshold = get_float(ocsort, "iou_threshold", cfg.ocsort.iou_threshold);
            cfg.ocsort.low_iou_threshold = get_float(ocsort, "low_iou_threshold", cfg.ocsort.low_iou_threshold);
            cfg.ocsort.inertia = get_float(ocsort, "inertia", cfg.ocsort.inertia);
            cfg.ocsort.delta_t = get_int(ocsort, "delta_t", cfg.ocsort.delta_t);
            cfg.ocsort.min_hits = get_int(ocsort, "min_hits", cfg.ocsort.min_hits);
            cfg.ocsort.max_age = get_int(ocsort, "max_age", cfg.ocsort.max_age);
            cfg.ocsort.min_box_area = get_float(ocsort, "min_box_area", cfg.ocsort.min_box_area);
            cfg.ocsort.use_byte = get_bool(ocsort, "use_byte", cfg.ocsort.use_byte);
        }
        return cfg;
    }

    static RecognizerModuleConfig parse_recognizer_module_config(const YAML::Node& n) {
        RecognizerModuleConfig cfg;
        if (!n) return cfg;

        cfg.type = get_str(n, "type", cfg.type);
        cfg.workers = get_int(n, "workers", cfg.workers);
        if (cfg.workers < 1) cfg.workers = 1;
        cfg.gallery_path = get_str(n, "gallery_path", cfg.gallery_path);
        cfg.unknown_threshold = n["unknown_threshold"] ? n["unknown_threshold"].as<float>() : cfg.unknown_threshold;
        return cfg;
    }

    static ModulesConfig parse_modules_config(const YAML::Node& n) {
        ModulesConfig cfg;
        if (!n) return cfg;

        cfg.detector = parse_detector_module_config(n["detector"]);
        cfg.tracker = parse_tracker_module_config(n["tracker"]);
        cfg.recognizer = parse_recognizer_module_config(n["recognizer"]);
        return cfg;
    }

    static MetricsConfig parse_metrics_config(const YAML::Node& n) {
        MetricsConfig cfg;
        if (!n) return cfg;

        cfg.enabled = get_bool(n, "enabled", cfg.enabled);
        cfg.enable_http = get_bool(n, "enable_http", cfg.enable_http);
        cfg.enable_ui_payload = get_bool(n, "enable_ui_payload", cfg.enable_ui_payload);
        cfg.log_interval_ms = get_int(n, "log_interval_ms", cfg.log_interval_ms);
        if (cfg.log_interval_ms < 250) cfg.log_interval_ms = 250;
        return cfg;
    }

    static ControllerConfig parse_controller_config(const YAML::Node& n) {
        ControllerConfig cfg;
        if (!n) return cfg;
        cfg.host = get_str(n, "host", cfg.host);
        cfg.port = get_int(n, "port", cfg.port);
        return cfg;
    }

    static RunnerGrpcConfig parse_runner_grpc_config(const YAML::Node& n) {
        RunnerGrpcConfig cfg;
        if (!n) return cfg;
        cfg.listen = get_str(n, "listen", cfg.listen);
        cfg.fallback_tcp = get_str(n, "fallback_tcp", cfg.fallback_tcp);
        return cfg;
    }

    static RunnerConfig parse_runner_config(const YAML::Node& n) {
        RunnerConfig cfg;
        if (!n) return cfg;
        cfg.id = get_str(n, "id", cfg.id);
        cfg.grpc = parse_runner_grpc_config(n["grpc"]);
        cfg.public_base_url = get_str(n, "public_base_url", cfg.public_base_url);
        return cfg;
    }

    static StreamingConfig parse_streaming_config(const YAML::Node& n) {
        StreamingConfig cfg;
        if (!n) return cfg;
        cfg.primary = get_str(n, "primary", cfg.primary);
        cfg.fallback = get_str(n, "fallback", cfg.fallback);
        cfg.codec = get_str(n, "codec", cfg.codec);
        cfg.encoder = get_str(n, "encoder", cfg.encoder);
        cfg.bitrate_kbps = get_int(n, "bitrate_kbps", cfg.bitrate_kbps);
        cfg.keyframe_interval_frames = get_int(n, "keyframe_interval_frames", cfg.keyframe_interval_frames);

        const YAML::Node wrtc = n["webrtc"];
        if (wrtc) {
            cfg.webrtc.enabled = get_bool(wrtc, "enabled", cfg.webrtc.enabled);
            cfg.webrtc.max_peers_per_stream = get_int(wrtc, "max_peers_per_stream", cfg.webrtc.max_peers_per_stream);
            cfg.webrtc.ice_gathering_timeout_ms = get_int(wrtc, "ice_gathering_timeout_ms", cfg.webrtc.ice_gathering_timeout_ms);
            cfg.webrtc.session_idle_timeout_s = get_int(wrtc, "session_idle_timeout_s", cfg.webrtc.session_idle_timeout_s);
            if (wrtc["stun_servers"]) cfg.webrtc.stun_servers = wrtc["stun_servers"].as<std::vector<std::string>>();
            if (wrtc["cors_allowed_origins"]) {
                cfg.webrtc.cors_allowed_origins = wrtc["cors_allowed_origins"].as<std::vector<std::string>>();
            }
        }
        return cfg;
    }

    static AppConfig parse_config_yaml_node(const YAML::Node& root) {
        AppConfig cfg;

        const YAML::Node srv = root["server"];
        cfg.server.url = get_str(srv, "host", "0.0.0.0");
        cfg.server.port = get_int(srv, "port", 8080);
        cfg.controller = parse_controller_config(root["controller"]);
        cfg.runner = parse_runner_config(root["runner"]);
        cfg.streaming = parse_streaming_config(root["streaming"]);
        cfg.modules = parse_modules_config(root["modules"]);
        cfg.metrics = parse_metrics_config(root["metrics"]);

        auto arr = root["streams"];
        if (!arr || !arr.IsSequence() || arr.size() <= 0) {
            throw std::runtime_error ("[Config] no streams specified!");
        }

        for (const auto& s : arr) {
            IngestConfig ic;
            ic.id = get_str(s, "id", "unk");
            ic.type = get_str(s, "type", "unk");

            ic.webcam = parse_webcam_config(s["webcam"]);
            ic.file = parse_file_config(s["file"]);
            ic.rtsp = parse_rtsp_config(s["rtsp"]);

            ic.replicate = parse_replicate_config(s["replicate"]);
            if (s["output"]) {
                throw std::runtime_error("[Config] stream.output is deprecated; use stream.outputs.profiles");
            }
            ic.output = parse_output_config(s["output"], OutputConfig{});
            ic.outputs = parse_outputs_config(s["outputs"]);
            if (ic.outputs.profiles.size() > 0 && ic.outputs.fps <= 0) {
                throw std::runtime_error("[Config] outputs.fps must be > 0 when outputs.profiles is configured");
            }

            if (ic.type == "rtsp" && ic.rtsp.url.empty()) {
                throw std::runtime_error ("[Config] RTSP stream " + ic.id + " has empty URL!");
            }

            cfg.streams.push_back(std::move(ic));
        }
        validate_config(cfg);
        return cfg;
    }

    AppConfig load_config_yaml(const std::string& path) {
        return parse_config_yaml_node(YAML::LoadFile(path));
    }

    AppConfig load_config_yaml_string(const std::string& yaml) {
        return parse_config_yaml_node(YAML::Load(yaml));
    }

    void validate_config(const AppConfig& config) {
        const auto& s = config.streaming;
        if (s.primary != "webrtc" && s.primary != "mjpeg") {
            throw std::runtime_error("[Config] streaming.primary must be 'webrtc' or 'mjpeg'");
        }
        if (s.fallback != "mjpeg" && s.fallback != "none") {
            throw std::runtime_error("[Config] streaming.fallback must be 'mjpeg' or 'none'");
        }
        if (s.codec != "h264") {
            throw std::runtime_error("[Config] streaming.codec must be 'h264'");
        }
        if (s.bitrate_kbps <= 0) {
            throw std::runtime_error("[Config] streaming.bitrate_kbps must be > 0");
        }
        if (s.keyframe_interval_frames <= 0) {
            throw std::runtime_error("[Config] streaming.keyframe_interval_frames must be > 0");
        }
        if (s.webrtc.max_peers_per_stream < 1) {
            throw std::runtime_error("[Config] streaming.webrtc.max_peers_per_stream must be >= 1");
        }
        if (s.webrtc.ice_gathering_timeout_ms < 250 || s.webrtc.ice_gathering_timeout_ms > 10000) {
            throw std::runtime_error("[Config] streaming.webrtc.ice_gathering_timeout_ms must be between 250 and 10000");
        }
        if (s.webrtc.session_idle_timeout_s < 1) {
            throw std::runtime_error("[Config] streaming.webrtc.session_idle_timeout_s must be >= 1");
        }
    }
}
