#include <anonymization/anonymizer.hpp>
#include <common/config.hpp>
#include <face_detector/face_detector.hpp>
#include <face_detector/face_policy.hpp>
#include <identity/identity_decider.hpp>
#include <person_detector/person_detector.hpp>
#include <pipeline/types.hpp>
#include <recognizer/recognizer.hpp>
#include <tracking/tracker.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace veilsight;

namespace {
    struct Args {
        std::string config_path;
        std::string video_path;
        std::string frames_dir;
        std::string sequence_id;
        std::string dataset = "ChokePoint";
        std::string system_id = "veilsight";
        std::string output_dir;
        std::string split_mode = "protected";
        std::string gallery_db;
        std::string attack_gallery_db;
        double deadline_ms = 40.0;
        double fps = 0.0;
        double source_fps = 30.0;
        bool video_only = false;
    };

    struct Stopwatch {
        using Clock = std::chrono::steady_clock;
        Clock::time_point t0 = Clock::now();

        double lap_ms() {
            const auto now = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - t0).count();
            t0 = now;
            return ms;
        }
    };

    std::string csv_escape(const std::string& value) {
        if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
        std::string out = "\"";
        for (char ch : value) {
            if (ch == '"') out += "\"\"";
            else out += ch;
        }
        out += '"';
        return out;
    }

    template <typename T>
    std::string to_cell(const T& value) {
        std::ostringstream ss;
        ss << value;
        return ss.str();
    }

    template <>
    std::string to_cell<std::string>(const std::string& value) {
        return csv_escape(value);
    }

    void write_row(std::ofstream& out, const std::vector<std::string>& cells) {
        for (size_t i = 0; i < cells.size(); ++i) {
            if (i) out << ',';
            out << csv_escape(cells[i]);
        }
        out << '\n';
    }

    std::string fmt_float(double value) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) << value;
        return ss.str();
    }

    void usage(const char* prog) {
        std::cerr << "Usage: " << prog << " --config <yaml> [--video <mp4>|--frames-dir <dir>] --sequence-id <id> "
                  << "--output-dir <dir> [--dataset ChokePoint] [--system-id veilsight] "
                  << "[--split-mode gallery|protected] [--gallery-db path] [--attack-gallery-db path] "
                  << "[--deadline-ms n] [--fps n] [--source-fps n] [--video-only]\n";
    }

    Args parse_args(int argc, char** argv) {
        Args args;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
                return argv[++i];
            };
            if (arg == "--config") args.config_path = require_value("--config");
            else if (arg == "--video") args.video_path = require_value("--video");
            else if (arg == "--frames-dir") args.frames_dir = require_value("--frames-dir");
            else if (arg == "--sequence-id") args.sequence_id = require_value("--sequence-id");
            else if (arg == "--dataset") args.dataset = require_value("--dataset");
            else if (arg == "--system-id") args.system_id = require_value("--system-id");
            else if (arg == "--output-dir") args.output_dir = require_value("--output-dir");
            else if (arg == "--split-mode") args.split_mode = require_value("--split-mode");
            else if (arg == "--gallery-db") args.gallery_db = require_value("--gallery-db");
            else if (arg == "--attack-gallery-db") args.attack_gallery_db = require_value("--attack-gallery-db");
            else if (arg == "--deadline-ms") args.deadline_ms = std::stod(require_value("--deadline-ms"));
            else if (arg == "--fps") args.fps = std::stod(require_value("--fps"));
            else if (arg == "--source-fps") args.source_fps = std::stod(require_value("--source-fps"));
            else if (arg == "--video-only") args.video_only = true;
            else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                std::exit(0);
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        }
        if (args.config_path.empty() || args.sequence_id.empty() || args.output_dir.empty()) {
            throw std::runtime_error("config, sequence-id, and output-dir are required");
        }
        if (args.video_path.empty() && args.frames_dir.empty()) {
            throw std::runtime_error("either --video or --frames-dir must be provided");
        }
        if (!args.video_path.empty() && !args.frames_dir.empty()) {
            throw std::runtime_error("provide either --video or --frames-dir, not both");
        }
        if (args.split_mode != "gallery" && args.split_mode != "protected") {
            throw std::runtime_error("split-mode must be gallery or protected");
        }
        if (args.fps < 0.0) {
            throw std::runtime_error("fps must be >= 0");
        }
        return args;
    }

    std::string frame_name(int64_t frame_id) {
        std::ostringstream ss;
        ss << std::setw(8) << std::setfill('0') << frame_id << ".jpg";
        return ss.str();
    }

    std::vector<fs::path> sorted_frame_files(const fs::path& dir) {
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            const auto& ext = entry.path().extension();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        return files;
    }

    int64_t filename_to_frame_id(const fs::path& p) {
        return std::stoll(p.stem().string());
    }

    std::string region_id(int64_t frame_id, int index) {
        return "r_" + std::to_string(frame_id) + "_" + std::to_string(index);
    }

    bool rect_contains_center(const RectF& face, const Box& body) {
        const float cx = face.x + face.w * 0.5f;
        const float cy = face.y + face.h * 0.5f;
        return cx >= body.x && cx <= body.x + body.w && cy >= body.y && cy <= body.y + body.h;
    }

    cv::Scalar track_color(const Box& track) {
        const std::string state = track.recognition_state;
        if (state == "known" || track.privacy_action == "allow") return cv::Scalar(87, 153, 21);
        if (state == "unknown") return cv::Scalar(34, 135, 216);
        if (state == "pending") return cv::Scalar(210, 125, 45);
        if (state == "failed") return cv::Scalar(87, 61, 195);
        return cv::Scalar(157, 147, 140);
    }

    cv::Rect clipped_rect(const cv::Mat& image, const RectF& rect) {
        cv::Rect out(
            static_cast<int>(std::lround(rect.x)),
            static_cast<int>(std::lround(rect.y)),
            static_cast<int>(std::lround(rect.w)),
            static_cast<int>(std::lround(rect.h)));
        out &= cv::Rect(0, 0, image.cols, image.rows);
        return out;
    }

    cv::Rect clipped_rect(const cv::Mat& image, const Box& box) {
        return clipped_rect(image, RectF{box.x, box.y, box.w, box.h});
    }

    void draw_label(cv::Mat& image,
                    const std::string& label,
                    const cv::Point& origin,
                    const cv::Scalar& color,
                    double font_scale = 0.42) {
        if (label.empty()) return;
        int baseline = 0;
        const int thickness = 1;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        const int x = std::clamp(origin.x, 0, std::max(0, image.cols - text_size.width - 6));
        const int y = std::clamp(origin.y, text_size.height + 6, std::max(text_size.height + 6, image.rows - 2));
        cv::Rect background(x, y - text_size.height - 5, text_size.width + 6, text_size.height + baseline + 6);
        background &= cv::Rect(0, 0, image.cols, image.rows);
        cv::rectangle(image, background, cv::Scalar(20, 24, 30), cv::FILLED);
        cv::putText(image,
                    label,
                    cv::Point(x + 3, y - 3),
                    cv::FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv::LINE_AA);
    }

    std::string track_label(const Box& track) {
        std::ostringstream ss;
        if (!track.identity_key.empty()) {
            ss << track.identity_key;
        } else {
            ss << "id:" << track.id;
        }
        if (!track.recognition_state.empty()) ss << " " << track.recognition_state;
        if (!track.identity_key.empty()) {
            ss << " " << std::fixed << std::setprecision(0) << (track.identity_confidence * 100.0f) << "%";
        }
        return ss.str();
    }

    void draw_track_overlay(cv::Mat& image, const std::vector<Box>& tracks) {
        if (image.empty()) return;
        for (const auto& track : tracks) {
            const cv::Rect box = clipped_rect(image, track);
            if (box.width < 2 || box.height < 2) continue;

            const cv::Scalar color = track_color(track);
            cv::rectangle(image, box, color, 2, cv::LINE_AA);
            const cv::Point foot(box.x + box.width / 2, box.y + box.height);
            cv::circle(image, foot, 4, color, cv::FILLED, cv::LINE_AA);
            draw_label(image, track_label(track), cv::Point(box.x + 4, box.y - 4), color);
        }
    }

    void draw_face_overlay(cv::Mat& image, const std::vector<Box>& tracks) {
        if (image.empty()) return;
        for (const auto& track : tracks) {
            if (!track.face) continue;
            const auto& face = *track.face;
            const cv::Rect face_box = clipped_rect(image, face.bbox);
            if (face_box.width < 2 || face_box.height < 2) continue;

            const cv::Scalar color = face.fresh ? cv::Scalar(255, 163, 73) : cv::Scalar(157, 147, 140);
            cv::rectangle(image, face_box, color, 2, cv::LINE_AA);
            draw_label(image,
                       "face " + fmt_float(face.score),
                       cv::Point(face_box.x + 3, face_box.y - 3),
                       color,
                       0.38);

            const int landmark_count = std::min(face.landmark_count, 5);
            for (int i = 0; i < landmark_count; ++i) {
                const PointF& lm = face.landmarks[static_cast<size_t>(i)];
                cv::circle(image,
                           cv::Point(static_cast<int>(std::lround(lm.x)), static_cast<int>(std::lround(lm.y))),
                           3,
                           color,
                           cv::FILLED,
                           cv::LINE_AA);
            }
        }
    }

    class ScenePriorOverlay {
    public:
        explicit ScenePriorOverlay(SceneGridConfig cfg)
            : rows_(std::max(1, cfg.rows)),
              cols_(std::max(1, cfg.cols)),
              decay_(std::clamp(cfg.occupancy_decay, 0.0f, 1.0f)),
              occupancy_(static_cast<size_t>(rows_ * cols_), 0.0f) {}

        void observe(int width, int height, const std::vector<Box>& tracks) {
            if (width <= 0 || height <= 0) return;
            for (float& value : occupancy_) value *= decay_;
            for (const auto& track : tracks) {
                if (track.id < 0) continue;
                const float x = std::clamp(track.x + track.w * 0.5f, 0.0f, static_cast<float>(width - 1));
                const float y = std::clamp(track.y + track.h * 0.95f, 0.0f, static_cast<float>(height - 1));
                int col = static_cast<int>(std::floor((x / static_cast<float>(width)) * cols_));
                int row = static_cast<int>(std::floor((y / static_cast<float>(height)) * rows_));
                col = std::clamp(col, 0, cols_ - 1);
                row = std::clamp(row, 0, rows_ - 1);
                occupancy_[static_cast<size_t>(row * cols_ + col)] += 1.0f;
            }
        }

        void draw(cv::Mat& image) const {
            if (image.empty() || occupancy_.empty()) return;
            const float max_value = *std::max_element(occupancy_.begin(), occupancy_.end());
            if (max_value <= 0.0f) return;

            cv::Mat heat = image.clone();
            const float cell_w = static_cast<float>(image.cols) / static_cast<float>(cols_);
            const float cell_h = static_cast<float>(image.rows) / static_cast<float>(rows_);
            for (int row = 0; row < rows_; ++row) {
                for (int col = 0; col < cols_; ++col) {
                    const float value = occupancy_[static_cast<size_t>(row * cols_ + col)];
                    if (value <= 0.01f) continue;
                    const float normalized = std::clamp(value / max_value, 0.0f, 1.0f);
                    cv::Rect cell(static_cast<int>(std::floor(col * cell_w)),
                                  static_cast<int>(std::floor(row * cell_h)),
                                  std::max(1, static_cast<int>(std::ceil(cell_w))),
                                  std::max(1, static_cast<int>(std::ceil(cell_h))));
                    cell &= cv::Rect(0, 0, image.cols, image.rows);
                    const cv::Scalar color(42, 125, 210 + 35 * normalized);
                    cv::rectangle(heat, cell, color, cv::FILLED);
                }
            }
            cv::addWeighted(heat, 0.24, image, 0.76, 0.0, image);

            const cv::Scalar grid_color(210, 220, 230);
            for (int row = 0; row <= rows_; row += std::max(1, rows_ / 6)) {
                const int y = static_cast<int>(std::round(row * cell_h));
                cv::line(image, cv::Point(0, y), cv::Point(image.cols, y), grid_color, 1, cv::LINE_AA);
            }
            for (int col = 0; col <= cols_; col += std::max(1, cols_ / 8)) {
                const int x = static_cast<int>(std::round(col * cell_w));
                cv::line(image, cv::Point(x, 0), cv::Point(x, image.rows), grid_color, 1, cv::LINE_AA);
            }
        }

    private:
        int rows_ = 1;
        int cols_ = 1;
        float decay_ = 0.92f;
        std::vector<float> occupancy_;
    };

    void ensure_video_writer(cv::VideoWriter& writer,
                             const fs::path& output_path,
                             double fps,
                             const cv::Size& size) {
        if (writer.isOpened()) return;
        fs::create_directories(output_path.parent_path());
        const double safe_fps = fps > 0.0 ? fps : 25.0;
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        if (!writer.open(output_path.string(), fourcc, safe_fps, size, true)) {
            throw std::runtime_error("failed to open video writer: " + output_path.string());
        }
    }
} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        usage(argv[0]);
        return 1;
    }

    AppConfig cfg;
    try {
        cfg = load_config_yaml(args.config_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    cfg.modules.face_detector.association_mode = "independent";
    cfg.modules.recognizer.gallery_path = args.gallery_db;
    const std::string attack_gallery_db = args.attack_gallery_db.empty() ? args.gallery_db : args.attack_gallery_db;
    auto attack_recognizer_cfg = cfg.modules.recognizer;
    attack_recognizer_cfg.gallery_path = attack_gallery_db;

    std::unique_ptr<IPersonDetector> person_detector;
    std::unique_ptr<ITracker> tracker;
    std::unique_ptr<IFaceDetector> face_detector;
    std::unique_ptr<IRecognizer> recognizer;
    std::unique_ptr<IRecognizer> attack_recognizer;
    std::unique_ptr<IIdentityDecider> identity;
    try {
        person_detector = create_person_detector(cfg.modules.person_detector);
        tracker = create_tracker(cfg.modules.tracker);
        face_detector = create_face_detector(cfg.modules.face_detector);
        recognizer = create_recognizer(cfg.modules.recognizer);
        if (!args.video_only) attack_recognizer = create_recognizer(attack_recognizer_cfg);
        identity = create_identity_decider(cfg.modules.identity);
    } catch (const std::exception& e) {
        std::cerr << "Model init error: " << e.what() << "\n";
        return 1;
    }

    AnonymizerConfig anon_cfg;
    anon_cfg.method = cfg.runtime.anonymizer.method;
    anon_cfg.pixelation_divisor = cfg.runtime.anonymizer.pixelation_divisor;
    anon_cfg.blur_kernel = cfg.runtime.anonymizer.blur_kernel;
    anon_cfg.face_only_when_available = cfg.runtime.anonymizer.face_only_when_available;
    Anonymizer anonymizer(anon_cfg);

    cv::VideoCapture cap;
    std::vector<fs::path> frame_files;
    int64_t frame_stride = 1;

    if (!args.frames_dir.empty()) {
        frame_files = sorted_frame_files(args.frames_dir);
        if (args.fps > 0.0 && args.source_fps > 0.0) {
            int64_t stride_val = static_cast<int64_t>(std::round(args.source_fps / args.fps));
            frame_stride = stride_val > 1 ? stride_val : 1;
        }
        std::cout << "Processing " << frame_files.size() << " frame files from "
                  << args.frames_dir << ", stride=" << frame_stride
                  << " (frame-id based)" << "\n";
    } else {
        cap.open(args.video_path);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video: " << args.video_path << "\n";
            return 1;
        }
        if (args.fps <= 0.0) {
            args.fps = cap.get(cv::CAP_PROP_FPS);
        }
        if (args.fps <= 0.0) {
            args.fps = 25.0;
        }
        std::cout << "Processing video " << args.video_path << " at " << args.fps << " fps\n";
    }

    const fs::path out_dir(args.output_dir);
    const fs::path frame_dir = out_dir / "output_frames";
    const fs::path mask_dir = out_dir / "masks";
    const fs::path anonymized_video_path = out_dir / "anonymized_tracker.mp4";
    const fs::path scene_prior_video_path = out_dir / "anonymized_tracker_scene_prior.mp4";
    const fs::path raw_video_path = out_dir / "raw_detections.mp4";
    fs::create_directories(out_dir);
    if (!args.video_only) {
        fs::create_directories(frame_dir);
        fs::create_directories(mask_dir);
    }

    std::ofstream face_log;
    std::ofstream anon_log;
    std::ofstream runtime_log;
    std::ofstream body_log;
    std::ofstream link_log;
    std::ofstream attack_log;
    if (!args.video_only) {
        face_log.open(out_dir / "face_log.csv");
        anon_log.open(out_dir / "anonymization_log.csv");
        runtime_log.open(out_dir / "frame_runtime_log.csv");
        body_log.open(out_dir / "body_log.csv");
        link_log.open(out_dir / "face_body_link_log.csv");
        attack_log.open(out_dir / "attack_log.csv");
        if (!face_log || !anon_log || !runtime_log || !body_log || !link_log || !attack_log) {
            std::cerr << "Cannot open output CSV files in " << out_dir << "\n";
            return 1;
        }

        write_row(face_log, {"system_id", "dataset", "sequence_id", "frame_id", "face_track_id", "face_det_id",
                             "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h", "face_confidence",
                             "face_size_px", "predicted_identity", "identity_confidence", "recognition_state",
                             "privacy_action", "anonymization_region_id", "output_frame_path"});
        write_row(anon_log, {"system_id", "dataset", "sequence_id", "frame_id", "region_id", "target_type",
                             "source_track_id", "source_track_type", "roi_x", "roi_y", "roi_w", "roi_h",
                             "method", "mask_path"});
        write_row(runtime_log, {"system_id", "dataset", "sequence_id", "frame_id", "input_frame_path",
                                "output_frame_path", "input_frame_received", "output_frame_emitted", "dropped_frame",
                                "latency_ms", "face_detector_ms", "face_tracker_ms", "body_detector_ms",
                                "body_tracker_ms", "recognizer_ms", "anonymizer_ms", "encoder_ms", "deadline_ms",
                                "deadline_missed", "error_message"});
        write_row(body_log, {"system_id", "dataset", "sequence_id", "frame_id", "body_track_id", "body_det_id",
                             "body_bbox_x", "body_bbox_y", "body_bbox_w", "body_bbox_h", "body_confidence",
                             "linked_face_track_id", "body_privacy_action", "anonymization_region_id",
                             "output_frame_path"});
        write_row(link_log, {"system_id", "dataset", "sequence_id", "frame_id", "face_track_id", "body_track_id",
                             "link_method", "face_inside_body", "link_score"});
        write_row(attack_log, {"system_id", "dataset", "sequence_id", "frame_id",
                               "face_det_id", "face_bbox_x", "face_bbox_y", "face_bbox_w",
                               "face_bbox_h", "face_confidence", "predicted_identity",
                               "identity_confidence", "recognized"});
    }

    std::ofstream config_json(out_dir / "config.json");
    config_json << "{\n"
                << "  \"system_id\": \"" << args.system_id << "\",\n"
                << "  \"dataset\": \"" << args.dataset << "\",\n"
                << "  \"sequence_id\": \"" << args.sequence_id << "\",\n"
                << "  \"split_mode\": \"" << args.split_mode << "\",\n"
                << "  \"video_path\": \"" << args.video_path << "\",\n"
                << "  \"frames_dir\": \"" << args.frames_dir << "\",\n"
                << "  \"fps\": " << fmt_float(args.fps) << ",\n"
                << "  \"source_fps\": " << fmt_float(args.source_fps) << ",\n"
                << "  \"anonymized_tracker_video\": \"" << anonymized_video_path.string() << "\",\n"
                << "  \"raw_detections_video\": \"" << raw_video_path.string() << "\",\n"
                << "  \"gallery_db\": \"" << args.gallery_db << "\",\n"
                << "  \"attack_gallery_db\": \"" << attack_gallery_db << "\"\n"
                << "}\n";

    HybridFacePolicy face_policy(cfg.modules.face_detector);
    FaceDetectorRunConfig detection_run_cfg;
    if (cfg.modules.face_detector.type == "yunet") {
        detection_run_cfg.input_w = cfg.modules.face_detector.yunet.input_w;
        detection_run_cfg.input_h = cfg.modules.face_detector.yunet.input_h;
    } else {
        detection_run_cfg.input_w = cfg.modules.face_detector.scrfd.input_w;
        detection_run_cfg.input_h = cfg.modules.face_detector.scrfd.input_h;
    }

    cv::VideoWriter anonymized_video;
    cv::VideoWriter scene_prior_video;
    cv::VideoWriter raw_video;
    ScenePriorOverlay scene_prior_overlay(cfg.modules.tracker.bytetrack.scene_grid);

    const auto process_frame = [&](cv::Mat& frame, int64_t fid) {
        const auto frame_start = Stopwatch::Clock::now();
        Stopwatch stage_timer;
        const fs::path output_path = frame_dir / frame_name(fid);
        std::string error_message;

        FramePtr ctx = std::make_shared<FrameCtx>();
        ctx->stream_id = args.sequence_id;
        ctx->source_type = "file";
        ctx->frame_id = fid;
        ctx->pts_ns = static_cast<int64_t>(static_cast<double>(fid) / args.fps * 1000000000.0);
        ctx->inf_w = frame.cols;
        ctx->inf_h = frame.rows;
        ctx->ui_w = frame.cols;
        ctx->ui_h = frame.rows;
        ctx->inf = frame.clone();
        ctx->ui = frame.clone();

        double body_detector_ms = 0.0;
        double body_tracker_ms = 0.0;
        double face_detector_ms = 0.0;
        double face_tracker_ms = 0.0;
        double recognizer_ms = 0.0;
        double anonymizer_ms = 0.0;
        double encoder_ms = 0.0;
        double attack_detect_ms = 0.0;
        double attack_recognize_ms = 0.0;

        std::vector<Box> tracks;
        std::map<int, std::string> region_by_track;
        try {
            const auto detections = person_detector->detect(ctx->inf);
            body_detector_ms = stage_timer.lap_ms();
            ctx->person_detection_count = detections.size();

            TrackerFrameInfo tracker_frame;
            tracker_frame.stream_id = args.sequence_id;
            tracker_frame.frame_id = fid;
            tracker_frame.width = frame.cols;
            tracker_frame.height = frame.rows;
            tracks = tracker->update(tracker_frame, detections);
            body_tracker_ms = stage_timer.lap_ms();

            face_policy.annotate(*ctx, tracks, *face_detector);
            face_detector_ms = stage_timer.lap_ms();
            face_tracker_ms = 0.0;
            ctx->face_detection_count = 0;
            for (const auto& track : tracks) {
                if (track.face) ++ctx->face_detection_count;
            }

            RecognitionTask recognition_task;
            recognition_task.stream_id = args.sequence_id;
            recognition_task.frame_id = fid;
            recognition_task.frame = ctx;
            recognition_task.tracks = tracks;
            RecognitionResult recognition_result = recognizer->recognize(recognition_task);
            recognizer_ms = stage_timer.lap_ms();

            IdentityTask identity_task;
            identity_task.stream_id = args.sequence_id;
            identity_task.frame_id = fid;
            identity_task.frame = ctx;
            identity_task.tracks = recognition_result.tracks;
            IdentityResult identity_result = identity->decide(identity_task);
            tracks = identity_result.tracks;
            ctx->tracked_boxes = tracks;

            if (!args.video_only) {
                cv::Mat raw_overlay = ctx->inf.clone();
                draw_track_overlay(raw_overlay, tracks);
                draw_face_overlay(raw_overlay, tracks);
                ensure_video_writer(raw_video, raw_video_path, args.fps, raw_overlay.size());
                raw_video.write(raw_overlay);
            }

            if (!args.video_only) {
                const auto regions = anonymizer.planned_regions(ctx->ui, tracks, 1.0f, 1.0f, 0.0f, 0.0f);
                for (size_t i = 0; i < regions.size(); ++i) {
                    const std::string rid = region_id(fid, static_cast<int>(i));
                    region_by_track[regions[i].source_track_id] = rid;
                    const fs::path mask_path = mask_dir / (rid + ".png");
                    cv::Mat mask = cv::Mat::zeros(ctx->ui.rows, ctx->ui.cols, CV_8UC1);
                    cv::rectangle(mask, regions[i].roi, cv::Scalar(255), cv::FILLED);
                    cv::imwrite(mask_path.string(), mask);
                    write_row(anon_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(fid), rid,
                                         regions[i].target_type, std::to_string(regions[i].source_track_id),
                                         regions[i].source_track_type, std::to_string(regions[i].roi.x),
                                         std::to_string(regions[i].roi.y), std::to_string(regions[i].roi.width),
                                         std::to_string(regions[i].roi.height), regions[i].method, mask_path.string()});
                }
            }
            anonymizer.apply(ctx->ui, tracks, 1.0f, 1.0f, 0.0f, 0.0f);
            anonymizer_ms = stage_timer.lap_ms();

            cv::Mat anonymized_overlay = ctx->ui.clone();
            draw_track_overlay(anonymized_overlay, tracks);
            draw_face_overlay(anonymized_overlay, tracks);
            ensure_video_writer(anonymized_video, anonymized_video_path, args.fps, anonymized_overlay.size());
            anonymized_video.write(anonymized_overlay);

            scene_prior_overlay.observe(ctx->ui.cols, ctx->ui.rows, tracks);
            cv::Mat scene_prior_overlay_frame = anonymized_overlay.clone();
            scene_prior_overlay.draw(scene_prior_overlay_frame);
            draw_track_overlay(scene_prior_overlay_frame, tracks);
            draw_face_overlay(scene_prior_overlay_frame, tracks);
            ensure_video_writer(scene_prior_video, scene_prior_video_path, args.fps, scene_prior_overlay_frame.size());
            scene_prior_video.write(scene_prior_overlay_frame);

            if (!args.video_only) {
                int face_det_id = 0;
                for (const auto& track : tracks) {
                    const std::string rid = region_by_track.count(track.id) ? region_by_track[track.id] : "";
                    if (track.face) {
                        const auto& face = *track.face;
                        const double face_size = std::max(face.bbox.w, face.bbox.h);
                        write_row(face_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(fid),
                                             std::to_string(track.id), std::to_string(face_det_id++),
                                             fmt_float(face.bbox.x), fmt_float(face.bbox.y), fmt_float(face.bbox.w),
                                             fmt_float(face.bbox.h), fmt_float(face.score), fmt_float(face_size),
                                             track.identity_key, fmt_float(track.identity_confidence),
                                             track.recognition_state, track.privacy_action, rid, output_path.string()});
                    }
                    if (track.id >= 0) {
                        const std::string linked_face = track.face ? std::to_string(track.id) : "";
                        write_row(body_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(fid),
                                             std::to_string(track.id), std::to_string(track.id), fmt_float(track.x),
                                             fmt_float(track.y), fmt_float(track.w), fmt_float(track.h), fmt_float(track.score),
                                             linked_face, track.privacy_action, rid, output_path.string()});
                        if (track.face) {
                            const bool inside = rect_contains_center(track.face->bbox, track);
                            write_row(link_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(fid),
                                                 std::to_string(track.id), std::to_string(track.id), "center_inside_body",
                                                 inside ? "1" : "0", inside ? "1.000" : "0.000"});
                        }
                    }
                }

                cv::imwrite(output_path.string(), ctx->ui);
            }
            encoder_ms = stage_timer.lap_ms();

            // Post-anonymization attack detection: run face detector + recognizer on anonymized output
            if (!args.video_only) try {
                const auto post_faces = face_detector->detect_faces(ctx->ui, detection_run_cfg);
                attack_detect_ms = stage_timer.lap_ms();
                int post_face_id = 0;
                for (const auto& post_face : post_faces) {
                    FramePtr attack_ctx = std::make_shared<FrameCtx>();
                    attack_ctx->stream_id = args.sequence_id;
                    attack_ctx->source_type = "file";
                    attack_ctx->frame_id = fid;
                    attack_ctx->inf_w = ctx->ui.cols;
                    attack_ctx->inf_h = ctx->ui.rows;
                    attack_ctx->ui_w = ctx->ui.cols;
                    attack_ctx->ui_h = ctx->ui.rows;
                    attack_ctx->inf = ctx->ui.clone();
                    attack_ctx->ui = ctx->ui.clone();

                    Box attack_track;
                    attack_track.id = -1;
                    attack_track.x = post_face.bbox.x;
                    attack_track.y = post_face.bbox.y;
                    attack_track.w = post_face.bbox.w;
                    attack_track.h = post_face.bbox.h;
                    attack_track.face = post_face;
                    std::vector<Box> attack_tracks = {attack_track};

                    RecognitionTask attack_rec_task;
                    attack_rec_task.stream_id = args.sequence_id;
                    attack_rec_task.frame_id = fid;
                    attack_rec_task.frame = attack_ctx;
                    attack_rec_task.tracks = attack_tracks;
                    RecognitionResult attack_result = attack_recognizer->recognize(attack_rec_task);

                    int recognized = 0;
                    std::string predicted_id;
                    float id_conf = 0.0f;
                    if (!attack_result.tracks.empty() &&
                        attack_result.tracks[0].recognition_state == "decided_known") {
                        recognized = 1;
                        predicted_id = attack_result.tracks[0].identity_key;
                        id_conf = attack_result.tracks[0].identity_confidence;
                    }
                    write_row(attack_log, {args.system_id, args.dataset, args.sequence_id,
                                           std::to_string(fid), std::to_string(post_face_id++),
                                           fmt_float(post_face.bbox.x), fmt_float(post_face.bbox.y),
                                           fmt_float(post_face.bbox.w), fmt_float(post_face.bbox.h),
                                           fmt_float(post_face.score), predicted_id,
                                           fmt_float(id_conf), std::to_string(recognized)});
                }
                attack_recognize_ms = stage_timer.lap_ms();
            } catch (const std::exception& e) {
                // Attack detection failure is non-fatal for the main pipeline
                std::cerr << "Attack detection error frame " << fid << ": " << e.what() << "\n";
            }

        } catch (const std::exception& e) {
            error_message = e.what();
        }

        const double latency_ms = std::chrono::duration<double, std::milli>(
                                      Stopwatch::Clock::now() - frame_start)
                                      .count();
        if (!args.video_only) {
            write_row(runtime_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(fid),
                                    args.video_path.empty()
                                        ? (args.frames_dir + ":" + std::to_string(fid))
                                        : (args.video_path + ":" + std::to_string(fid)),
                                    output_path.string(), "1",
                                    error_message.empty() ? "1" : "0", "0", fmt_float(latency_ms),
                                    fmt_float(face_detector_ms), fmt_float(face_tracker_ms), fmt_float(body_detector_ms),
                                    fmt_float(body_tracker_ms), fmt_float(recognizer_ms), fmt_float(anonymizer_ms),
                                    fmt_float(encoder_ms), fmt_float(args.deadline_ms),
                                    latency_ms > args.deadline_ms ? "1" : "0", error_message});
        }
        if (!error_message.empty()) {
            std::cerr << "Frame " << fid << " error: " << error_message << "\n";
        }
    };

    // Process frames
    int64_t processed_count = 0;
    if (!args.frames_dir.empty()) {
        for (const auto& filepath : frame_files) {
            int64_t fid = filename_to_frame_id(filepath);
            if (fid % frame_stride != 0) continue;
            cv::Mat frame = cv::imread(filepath.string());
            if (frame.empty()) {
                std::cerr << "Cannot read frame: " << filepath << "\n";
                continue;
            }
            process_frame(frame, fid);
            ++processed_count;
        }
    } else {
        int64_t frame_id = 0;
        cv::Mat frame;
        while (cap.read(frame)) {
            process_frame(frame, frame_id);
            ++frame_id;
            ++processed_count;
        }
    }

    if (anonymized_video.isOpened()) anonymized_video.release();
    if (scene_prior_video.isOpened()) scene_prior_video.release();
    if (raw_video.isOpened()) raw_video.release();

    std::cout << "Processed " << processed_count << " frames into " << out_dir << "\n";
    std::cout << "Wrote anonymized tracker video: " << anonymized_video_path << "\n";
    std::cout << "Wrote scene-prior anonymized tracker video: " << scene_prior_video_path << "\n";
    if (!args.video_only) std::cout << "Wrote raw detections video: " << raw_video_path << "\n";
    return 0;
}
