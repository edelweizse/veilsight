#include <common/config.hpp>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace ss {
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

        cfg.model_path = get_str(n, "model_path", cfg.model_path);
        cfg.input_w = get_int(n, "input_w", cfg.input_w);
        cfg.input_h = get_int(n, "input_h", cfg.input_h);
        cfg.score_threshold = get_float(n, "score_threshold", cfg.score_threshold);
        cfg.nms_threshold = get_float(n, "nms_threshold", cfg.nms_threshold);
        cfg.top_k = get_int(n, "top_k", cfg.top_k);
        cfg.person_class_id = get_int(n, "person_class_id", cfg.person_class_id);
        cfg.letterbox = get_bool(n, "letterbox", cfg.letterbox);
        cfg.decoded_output = get_bool(n, "decoded_output", cfg.decoded_output);
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

    AppConfig load_config_yaml(const std::string& path) {
        AppConfig cfg;
        YAML::Node root = YAML::LoadFile(path);

        const YAML::Node srv = root["server"];
        cfg.server.url = get_str(srv, "host", "0.0.0.0");
        cfg.server.port = get_int(srv, "port", 8080);
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
            ic.output = parse_output_config(s["output"], OutputConfig{});
            ic.outputs = parse_outputs_config(s["outputs"]);

            if (ic.type == "rtsp" && ic.rtsp.url.empty()) {
                throw std::runtime_error ("[Config] RTSP stream " + ic.id + " has empty URL!");
            }

            cfg.streams.push_back(std::move(ic));
        }
        return cfg;
    }
}
