#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace veilsight {
    struct YuNetModuleConfig {
        std::string param_path = "models/detector/yunet/face_detection_yunet_2023mar.ncnn.param";
        std::string bin_path = "models/detector/yunet/face_detection_yunet_2023mar.ncnn.bin";
        int input_w = 1088;
        int input_h = 608;
        float score_threshold = 0.6f;
        float nms_threshold = 0.3f;
        int top_k = 750;
        int ncnn_threads = 1;
    };

    struct SCRFDModuleConfig {
        std::string variant = "500m";
        std::string param_path = "models/detector/scrfd_500m/scrfd_500m.ncnn.param";
        std::string bin_path = "models/detector/scrfd_500m/scrfd_500m.ncnn.bin";
        int input_w = 640;
        int input_h = 640;
        float score_threshold = 0.35f;
        float nms_threshold = 0.3f;
        int top_k = 750;
        int ncnn_threads = 1;
    };

    struct YoloXModuleConfig {
        std::string variant = "nano";
        std::string model_path = "models/detector/bytetrack_nano";
        std::string param_path = "models/detector/bytetrack_nano.ncnn.param";
        std::string bin_path = "models/detector/bytetrack_nano.ncnn.bin";
        int input_w = 640;
        int input_h = 640;
        float score_threshold = 0.35f;
        float nms_threshold = 0.45f;
        int top_k = 300;
        int class_id = 0;
        int ncnn_threads = 1;
        bool letterbox = true;
        bool decoded_output = false;
    };

    struct DetectorModuleConfig {
        std::string type = "yolox"; // yolox|yunet|scrfd
        int workers = 1;
        YuNetModuleConfig yunet;
        SCRFDModuleConfig scrfd;
        YoloXModuleConfig yolox;
    };

    struct DemoTrackerModuleConfig {
        float high_thresh = 0.6f;
        float low_thresh = 0.2f;
        float match_iou_thresh = 0.3f;
        float low_match_iou_thresh = 0.2f;
        int min_hits = 2;
        int max_missed = 20;
    };

    struct SceneGridConfig {
        bool enabled = true;
        int rows = 6;
        int cols = 8;
        float association_weight = 0.15f;
        float cell_distance_weight = 0.65f;
        float occupancy_weight = 0.15f;
        float transition_weight = 0.20f;
        float max_extra_cost = 0.30f;
        float occupancy_decay = 0.92f;
        float transition_decay = 0.98f;
        int warmup_frames = 5;
    };

    struct ByteTrackModuleConfig {
        float high_thresh = 0.45f;
        float low_thresh = 0.2f;
        float new_track_thresh = 0.45f;
        float match_iou_thresh = 0.35f;
        float low_match_iou_thresh = 0.25f;
        float unconfirmed_match_iou_thresh = 0.35f;
        float duplicate_iou_thresh = 0.85f;
        int track_buffer = 100;
        float min_box_area = 0.0f;
        bool fuse_score = true;
        SceneGridConfig scene_grid;
    };

    struct OCSortModuleConfig {
        float det_thresh = 0.5f;
        float low_det_thresh = 0.1f;
        float iou_threshold = 0.3f;
        float low_iou_threshold = 0.2f;
        float inertia = 0.2f;
        int delta_t = 3;
        int min_hits = 3;
        int max_age = 30;
        float min_box_area = 10.0f;
        bool use_byte = true;
    };

    struct TrackerModuleConfig {
        std::string type = "demo"; // demo|bytetrack|ocsort
        DemoTrackerModuleConfig demo;
        ByteTrackModuleConfig bytetrack;
        OCSortModuleConfig ocsort;
    };

    struct RecognizerModuleConfig {
        std::string type = "none"; // none
        int workers = 1;
        std::string gallery_path;
        float unknown_threshold = 0.0f;
    };

    struct ModulesConfig {
        DetectorModuleConfig detector;
        TrackerModuleConfig tracker;
        RecognizerModuleConfig recognizer;
    };

    struct MetricsConfig {
        bool enabled = true;
        bool enable_http = true;
        bool enable_ui_payload = true;
        int log_interval_ms = 5000;
    };

    struct WebcamConfig {
        std::string device;
        int width;
        int height;
        //int fps;
        bool mjpg;
    };

    struct FileConfig {
        std::string path;
        int fps;
        bool loop;
    };

    struct RTSPConfig {
        std::string url;
        int latency_ms;
        //int fps;
        bool tcp;
    };

    struct ReplicateConfig {
        int count = 1;
        std::vector<std::string> ids;
    };

    struct OutputConfig {
        int width = 0;
        int height = 0;
        int fps = 0;
        bool keep_aspect = true;
        std::string interp = "linear"; // nearest|cubic|linear|area
        std::string format = "BGR";
        int jpeg_quality = 75;
    };

    struct OutputsConfig {
        int fps = 0;
        std::unordered_map<std::string, OutputConfig> profiles;
    };

    struct IngestConfig {
        std::string type; // webcam|file|rtsp
        std::string id;

        WebcamConfig webcam;
        FileConfig file;
        RTSPConfig rtsp;

        ReplicateConfig replicate;

        OutputConfig output;

        OutputsConfig outputs;
    };

    struct ServerConfig {
        std::string url = "0.0.0.0";
        int port = 8080;
    };

    struct ControllerConfig {
        std::string host = "0.0.0.0";
        int port = 8000;
    };

    struct RunnerGrpcConfig {
        std::string listen = "unix:///tmp/veilsight-runner.sock";
        std::string fallback_tcp = "127.0.0.1:9090";
    };

    struct RunnerConfig {
        std::string id = "edge-0";
        RunnerGrpcConfig grpc;
        std::string public_base_url = "http://localhost:8080";
    };

    struct StreamingConfig {
        struct WebRTCConfig {
            bool enabled = true;
            int max_peers_per_stream = 2;
            int ice_gathering_timeout_ms = 2000;
            int session_idle_timeout_s = 30;
            std::vector<std::string> stun_servers;
            std::vector<std::string> cors_allowed_origins = {
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            };
        };

        std::string primary = "webrtc";
        std::string fallback = "mjpeg";
        std::string codec = "h264";
        std::string encoder = "auto";
        int bitrate_kbps = 2500;
        int keyframe_interval_frames = 30;
        WebRTCConfig webrtc;
    };

    struct AppConfig {
        ServerConfig server;
        ControllerConfig controller;
        RunnerConfig runner;
        StreamingConfig streaming;
        ModulesConfig modules;
        MetricsConfig metrics;
        std::vector<IngestConfig> streams;
    };

    AppConfig load_config_yaml(const std::string& path);
    AppConfig load_config_yaml_string(const std::string& yaml);
    void validate_config(const AppConfig& config);
}
