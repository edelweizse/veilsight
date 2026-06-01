#include <anonymization/anonymizer.hpp>
#include <common/config.hpp>
#include <face_detector/face_detector.hpp>
#include <face_detector/face_policy.hpp>
#include <identity/identity_decider.hpp>
#include <person_detector/person_detector.hpp>
#include <pipeline/types.hpp>
#include <recognizer/recognizer.hpp>
#include <render/rules.hpp>
#include <tracking/tracker.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace veilsight;

namespace {
    struct Args {
        bool preview_frame = false;
        bool overwrite = false;
        std::string config_path;
        std::string input_path;
        std::string output_path;
        std::string gallery_db;
        std::string layers_raw;
        std::string rules_yaml;
        std::string events_jsonl;
        std::string manifest_path;
        std::string preview_jpeg;
        std::string stream_id;
        std::string timing_mode = "source";
        double fps = 0.0;
        double source_fps = 0.0;
        int preview_every_frames = 5;
        bool no_gallery = false;
        bool fps_provided = false;
        std::string mode = "face+body";

        bool is_face_only() const { return mode == "face"; }
    };

    void usage(const char* prog) {
        std::cerr
            << "Usage:\n"
            << "  " << prog << " --preview-frame --input <video> --output <jpg>\n"
            << "  " << prog << " --config <yaml> --input <video> --output <mp4> "
            << "[--gallery-db path] [--layers tracks,faces,directions,rules,events] "
            << "[--rules-yaml path] [--events-jsonl path] [--manifest path] "
            << "[--preview-jpeg path] [--preview-every-frames n] [--no-gallery] "
            << "[--timing-mode source|custom] [--fps n] [--source-fps n] [--stream-id id] [--overwrite] "
            << "[--mode face|face+body]\n";
    }

    Args parse_args(int argc, char** argv) {
        Args args;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
                return argv[++i];
            };
            if (arg == "--preview-frame") args.preview_frame = true;
            else if (arg == "--overwrite") args.overwrite = true;
            else if (arg == "--config") args.config_path = require_value("--config");
            else if (arg == "--input") args.input_path = require_value("--input");
            else if (arg == "--output") args.output_path = require_value("--output");
            else if (arg == "--gallery-db") args.gallery_db = require_value("--gallery-db");
            else if (arg == "--layers") args.layers_raw = require_value("--layers");
            else if (arg == "--rules-yaml") args.rules_yaml = require_value("--rules-yaml");
            else if (arg == "--events-jsonl") args.events_jsonl = require_value("--events-jsonl");
            else if (arg == "--manifest") args.manifest_path = require_value("--manifest");
            else if (arg == "--preview-jpeg") args.preview_jpeg = require_value("--preview-jpeg");
            else if (arg == "--preview-every-frames") args.preview_every_frames = std::stoi(require_value("--preview-every-frames"));
            else if (arg == "--no-gallery") args.no_gallery = true;
            else if (arg == "--timing-mode") args.timing_mode = require_value("--timing-mode");
            else if (arg == "--fps") {
                args.fps = std::stod(require_value("--fps"));
                args.fps_provided = true;
            }
            else if (arg == "--source-fps") args.source_fps = std::stod(require_value("--source-fps"));
            else if (arg == "--stream-id") args.stream_id = require_value("--stream-id");
            else if (arg == "--mode") args.mode = require_value("--mode");
            else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                std::exit(0);
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        }

        if (args.input_path.empty() || args.output_path.empty()) {
            throw std::runtime_error("input and output are required");
        }
        if (!args.preview_frame && args.config_path.empty()) {
            throw std::runtime_error("config is required for render mode");
        }
        if (args.fps < 0.0 || args.source_fps < 0.0) {
            throw std::runtime_error("fps and source-fps must be >= 0");
        }
        if (args.timing_mode != "source" && args.timing_mode != "custom") {
            throw std::runtime_error("timing-mode must be source or custom");
        }
        if (args.timing_mode == "custom" && (!args.fps_provided || args.fps <= 0.0)) {
            throw std::runtime_error("--fps > 0 is required when --timing-mode custom");
        }
        if (args.preview_every_frames < 1) {
            throw std::runtime_error("preview-every-frames must be >= 1");
        }
        if (args.mode != "face" && args.mode != "face+body") {
            throw std::runtime_error("mode must be face or face+body");
        }
        return args;
    }

    void create_parent(const fs::path& path) {
        const fs::path parent = path.parent_path();
        if (!parent.empty()) fs::create_directories(parent);
    }

    std::string stem_for_stream_id(const fs::path& input_path) {
        const std::string stem = input_path.stem().string();
        return stem.empty() ? "render" : stem;
    }

    std::set<std::string> parse_layers(const std::string& raw) {
        std::set<std::string> layers;
        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ',')) {
            item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
            item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), item.end());
            if (!item.empty()) layers.insert(item);
        }
        return layers;
    }

    cv::Scalar track_color(const Box& track) {
        if (track.recognition_state == "known" || track.privacy_action == "allow") return cv::Scalar(87, 153, 21);
        if (track.recognition_state == "unknown") return cv::Scalar(34, 135, 216);
        if (track.recognition_state == "pending") return cv::Scalar(210, 125, 45);
        if (track.recognition_state == "failed") return cv::Scalar(87, 61, 195);
        return cv::Scalar(157, 147, 140);
    }

    cv::Rect clipped_rect(const cv::Mat& image, const RectF& rect) {
        cv::Rect out(static_cast<int>(std::lround(rect.x)),
                     static_cast<int>(std::lround(rect.y)),
                     static_cast<int>(std::lround(rect.w)),
                     static_cast<int>(std::lround(rect.h)));
        out &= cv::Rect(0, 0, image.cols, image.rows);
        return out;
    }

    cv::Rect clipped_rect(const cv::Mat& image, const Box& box) {
        return clipped_rect(image, RectF{box.x, box.y, box.w, box.h});
    }

    std::string fmt_float(double value, int precision = 3) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(precision) << value;
        return ss.str();
    }

    void draw_label(cv::Mat& image, const std::string& label, const cv::Point& origin, const cv::Scalar& color) {
        if (label.empty()) return;
        int baseline = 0;
        const double font_scale = 0.42;
        const int thickness = 1;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        const int x = std::clamp(origin.x, 0, std::max(0, image.cols - text_size.width - 6));
        const int y = std::clamp(origin.y, text_size.height + 6, std::max(text_size.height + 6, image.rows - 2));
        cv::Rect background(x, y - text_size.height - 5, text_size.width + 6, text_size.height + baseline + 6);
        background &= cv::Rect(0, 0, image.cols, image.rows);
        cv::rectangle(image, background, cv::Scalar(20, 24, 30), cv::FILLED);
        cv::putText(image, label, cv::Point(x + 3, y - 3), cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv::LINE_AA);
    }

    void draw_tracks(cv::Mat& image, const std::vector<Box>& tracks) {
        for (const Box& track : tracks) {
            const cv::Rect box = clipped_rect(image, track);
            if (box.width < 2 || box.height < 2) continue;
            const cv::Scalar color = track_color(track);
            cv::rectangle(image, box, color, 2, cv::LINE_AA);
            cv::circle(image, cv::Point(box.x + box.width / 2, box.y + box.height), 4, color, cv::FILLED, cv::LINE_AA);

            std::ostringstream label;
            label << (track.identity_key.empty() ? "id:" + std::to_string(track.id) : track.identity_key);
            if (!track.recognition_state.empty()) label << " " << track.recognition_state;
            if (!track.identity_key.empty()) label << " " << fmt_float(track.identity_confidence * 100.0, 0) << "%";
            draw_label(image, label.str(), cv::Point(box.x + 4, box.y - 4), color);
        }
    }

    void draw_faces(cv::Mat& image, const std::vector<Box>& tracks) {
        for (const Box& track : tracks) {
            if (!track.face) continue;
            const FaceObservation& face = *track.face;
            const cv::Rect face_box = clipped_rect(image, face.bbox);
            if (face_box.width < 2 || face_box.height < 2) continue;
            const cv::Scalar color = face.fresh ? cv::Scalar(255, 163, 73) : cv::Scalar(157, 147, 140);
            cv::rectangle(image, face_box, color, 2, cv::LINE_AA);
            draw_label(image, "face " + fmt_float(face.score, 2), cv::Point(face_box.x + 3, face_box.y - 3), color);
            for (int i = 0; i < std::min(face.landmark_count, 5); ++i) {
                const PointF& p = face.landmarks[static_cast<size_t>(i)];
                cv::circle(image,
                           cv::Point(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y))),
                           3,
                           color,
                           cv::FILLED,
                           cv::LINE_AA);
            }
        }
    }

    PointF foot_point(const Box& track) {
        return PointF{track.x + track.w * 0.5f, track.y + track.h};
    }

    void draw_directions(cv::Mat& image, const std::vector<Box>& tracks, const std::map<int, PointF>& previous) {
        const cv::Scalar color(40, 140, 242);
        for (const Box& track : tracks) {
            if (track.id < 0) continue;
            const auto it = previous.find(track.id);
            if (it == previous.end()) continue;
            const PointF from = it->second;
            const PointF to = foot_point(track);
            if (std::hypot(to.x - from.x, to.y - from.y) < 2.0f) continue;
            cv::Point p1(static_cast<int>(std::lround(from.x)), static_cast<int>(std::lround(from.y)));
            cv::Point p2(static_cast<int>(std::lround(to.x)), static_cast<int>(std::lround(to.y)));
            cv::arrowedLine(image, p1, p2, color, 2, cv::LINE_AA, 0, 0.28);
        }
    }

    void draw_rules(cv::Mat& image, const std::vector<veilsight::render::RenderRule>& rules) {
        for (const auto& rule : rules) {
            if (!rule.enabled || rule.points.empty()) continue;
            std::vector<cv::Point> points;
            points.reserve(rule.points.size());
            for (const PointF& point : rule.points) {
                points.emplace_back(static_cast<int>(std::lround(point.x)), static_cast<int>(std::lround(point.y)));
            }
            const cv::Scalar color(51, 204, 255);
            if (rule.kind == "area" && points.size() >= 3) {
                std::vector<std::vector<cv::Point>> polys{points};
                cv::Mat fill = image.clone();
                cv::fillPoly(fill, polys, cv::Scalar(51, 204, 255));
                cv::addWeighted(fill, 0.16, image, 0.84, 0.0, image);
                cv::polylines(image, polys, true, color, 2, cv::LINE_AA);
            } else if (rule.kind == "line" && points.size() >= 2) {
                cv::line(image, points[0], points[1], color, 2, cv::LINE_AA);
            }
            if (!points.empty()) draw_label(image, rule.name, points.front() + cv::Point(7, -7), color);
        }
    }

    void draw_events(cv::Mat& image, const std::vector<veilsight::render::RenderEvent>& events) {
        const cv::Scalar color(87, 61, 195);
        for (const auto& event : events) {
            cv::Point p(static_cast<int>(std::lround(event.position.x)), static_cast<int>(std::lround(event.position.y)));
            cv::circle(image, p, 7, color, cv::FILLED, cv::LINE_AA);
            draw_label(image, event.kind, p + cv::Point(8, 4), color);
        }
    }

    std::string json_escape(const std::string& value) {
        std::ostringstream out;
        for (char ch : value) {
            switch (ch) {
                case '\\': out << "\\\\"; break;
                case '"': out << "\\\""; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default: out << ch; break;
            }
        }
        return out.str();
    }

    void write_manifest(const fs::path& path,
                        const Args& args,
                        int64_t frames,
                        double fps,
                        double source_fps,
                        const std::set<std::string>& layers,
                        const fs::path& events_path) {
        create_parent(path);
        std::ofstream out(path);
        if (!out) throw std::runtime_error("failed to open manifest: " + path.string());
        out << "{\n";
        out << "  \"input_path\": \"" << json_escape(fs::absolute(args.input_path).string()) << "\",\n";
        out << "  \"output_path\": \"" << json_escape(fs::absolute(args.output_path).string()) << "\",\n";
        out << "  \"events_jsonl\": \"" << json_escape(fs::absolute(events_path).string()) << "\",\n";
        out << "  \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
        out << "  \"no_gallery\": " << (args.no_gallery ? "true" : "false") << ",\n";
        out << "  \"preview_jpeg\": \"" << json_escape(args.preview_jpeg.empty() ? "" : fs::absolute(args.preview_jpeg).string()) << "\",\n";
        out << "  \"frames\": " << frames << ",\n";
        out << "  \"fps\": " << fmt_float(fps) << ",\n";
        out << "  \"source_fps\": " << fmt_float(source_fps) << ",\n";
        out << "  \"stream_id\": \"" << json_escape(args.stream_id) << "\",\n";
        out << "  \"layers\": [";
        bool first = true;
        for (const std::string& layer : layers) {
            if (!first) out << ", ";
            first = false;
            out << "\"" << json_escape(layer) << "\"";
        }
        out << "]\n";
        out << "}\n";
    }

    int preview_frame(const Args& args) {
        cv::VideoCapture cap(args.input_path);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open input video: " << args.input_path << "\n";
            return 1;
        }
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Input video has no readable frames: " << args.input_path << "\n";
            return 1;
        }
        const fs::path out(args.output_path);
        create_parent(out);
        if (!cv::imwrite(out.string(), frame)) {
            std::cerr << "Failed to write preview frame: " << out << "\n";
            return 1;
        }
        return 0;
    }

    bool write_preview_jpeg(const fs::path& preview_path, const cv::Mat& image) {
        if (preview_path.empty()) return true;
        create_parent(preview_path);
        const fs::path tmp_path(preview_path.string() + ".tmp.jpg");
        if (!cv::imwrite(tmp_path.string(), image)) return false;
        std::error_code ec;
        fs::rename(tmp_path, preview_path, ec);
        if (ec) {
            fs::remove(preview_path, ec);
            ec.clear();
            fs::rename(tmp_path, preview_path, ec);
        }
        return !ec;
    }
}

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        usage(argv[0]);
        return 1;
    }

    if (args.preview_frame) return preview_frame(args);

    const fs::path output_path(args.output_path);
    if (fs::exists(output_path) && !args.overwrite) {
        std::cerr << "Output exists; pass --overwrite to replace it: " << output_path << "\n";
        return 1;
    }

    cv::VideoCapture cap(args.input_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open input video: " << args.input_path << "\n";
        return 1;
    }
    const double metadata_fps = cap.get(cv::CAP_PROP_FPS);
    double effective_fps = args.timing_mode == "custom" ? args.fps : metadata_fps;
    if (effective_fps <= 0.0) effective_fps = 25.0;
    const double source_fps = args.timing_mode == "custom"
                                  ? (args.source_fps > 0.0 ? args.source_fps : metadata_fps)
                                  : effective_fps;
    const int64_t total_frames = static_cast<int64_t>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (args.stream_id.empty()) args.stream_id = stem_for_stream_id(args.input_path);

    AppConfig cfg;
    try {
        cfg = load_config_yaml(args.config_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }
    if (args.no_gallery) {
        cfg.modules.recognizer.type = "noop";
        cfg.modules.recognizer.gallery_path.clear();
        cfg.modules.identity.type = "noop";
        cfg.modules.identity.gallery_path.clear();
    } else if (!args.gallery_db.empty()) {
        cfg.modules.recognizer.gallery_path = args.gallery_db;
    }

    std::unique_ptr<IPersonDetector> person_detector;
    std::unique_ptr<ITracker> tracker;
    std::unique_ptr<IFaceDetector> face_detector;
    std::unique_ptr<IRecognizer> recognizer;
    std::unique_ptr<IIdentityDecider> identity;
    try {
        person_detector = create_person_detector(cfg.modules.person_detector);
        tracker = create_tracker(cfg.modules.tracker);
        face_detector = create_face_detector(cfg.modules.face_detector);
        if (!args.no_gallery) {
            recognizer = create_recognizer(cfg.modules.recognizer);
            identity = create_identity_decider(cfg.modules.identity);
        }
    } catch (const std::exception& e) {
        std::cerr << "Model init error: " << e.what() << "\n";
        return 1;
    }

    AnonymizerConfig anon_cfg;
    anon_cfg.method = cfg.runtime.anonymizer.method;
    anon_cfg.pixelation_divisor = cfg.runtime.anonymizer.pixelation_divisor;
    anon_cfg.blur_kernel = cfg.runtime.anonymizer.blur_kernel;
    anon_cfg.face_only_when_available = cfg.runtime.anonymizer.face_only_when_available;
    if (args.is_face_only()) {
        anon_cfg.face_only_when_available = true;
        anon_cfg.strict_face_only = true;
    }
    Anonymizer anonymizer(anon_cfg);

    std::vector<veilsight::render::RenderRule> rules;
    try {
        rules = veilsight::render::load_render_rules_yaml(args.rules_yaml);
    } catch (const std::exception& e) {
        std::cerr << "Rules error: " << e.what() << "\n";
        return 1;
    }
    veilsight::render::RenderRuleEngine rule_engine(rules);
    const std::set<std::string> layers = parse_layers(args.layers_raw);

    const fs::path events_path = args.events_jsonl.empty()
                                     ? output_path.parent_path() / (output_path.stem().string() + ".events.jsonl")
                                     : fs::path(args.events_jsonl);
    const fs::path manifest_path = args.manifest_path.empty()
                                       ? output_path.parent_path() / (output_path.stem().string() + ".manifest.json")
                                       : fs::path(args.manifest_path);

    create_parent(output_path);
    create_parent(events_path);
    std::ofstream events_out(events_path);
    if (!events_out) {
        std::cerr << "Failed to open events JSONL: " << events_path << "\n";
        return 1;
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Input video has no readable frames: " << args.input_path << "\n";
        return 1;
    }

    cv::VideoWriter writer;
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    if (!writer.open(output_path.string(), fourcc, effective_fps, frame.size(), true)) {
        std::cerr << "Failed to open output MP4: " << output_path << "\n";
        return 1;
    }

    if (args.is_face_only()) {
        cfg.modules.face_detector.association_mode = "independent";
        face_detector = create_face_detector(cfg.modules.face_detector);
    }
    HybridFacePolicy face_policy(cfg.modules.face_detector);
    std::map<int, PointF> previous_positions;
    std::vector<veilsight::render::RenderEvent> recent_events;
    const fs::path preview_jpeg_path(args.preview_jpeg);

    int64_t frame_id = 0;
    do {
        FramePtr ctx = std::make_shared<FrameCtx>();
        ctx->stream_id = args.stream_id;
        ctx->source_type = "file";
        ctx->frame_id = frame_id;
        ctx->pts_ns = static_cast<int64_t>(static_cast<double>(frame_id) / source_fps * 1000000000.0);
        ctx->inf_w = frame.cols;
        ctx->inf_h = frame.rows;
        ctx->ui_w = frame.cols;
        ctx->ui_h = frame.rows;
        ctx->inf = frame.clone();
        ctx->ui = frame.clone();

        std::vector<Box> tracks;
        try {
            if (args.is_face_only()) {
                ctx->person_detection_count = 0;
                ctx->inf = frame.clone();
                if (face_detector) {
                    face_policy.annotate(*ctx, tracks, *face_detector);
                }
                ctx->face_detection_count = tracks.size();
                // Expand face-only boxes so anonymization is visible.
                for (auto& track : tracks) {
                    const float cx = track.x + track.w * 0.5f;
                    const float cy = track.y + track.h * 0.5f;
                    constexpr float kExpand = 4.0f;
                    const float ew = track.w * kExpand;
                    const float eh = track.h * kExpand;
                    track.x = cx - ew * 0.5f;
                    track.y = cy - eh * 0.5f;
                    track.w = ew;
                    track.h = eh;
                    if (track.face) {
                        track.face->bbox.x = track.x;
                        track.face->bbox.y = track.y;
                        track.face->bbox.w = ew;
                        track.face->bbox.h = eh;
                    }
                }
            } else {
                const auto detections = person_detector->detect(ctx->inf);
                ctx->person_detection_count = detections.size();

                TrackerFrameInfo tracker_frame;
                tracker_frame.stream_id = args.stream_id;
                tracker_frame.frame_id = frame_id;
                tracker_frame.width = frame.cols;
                tracker_frame.height = frame.rows;
                tracks = tracker->update(tracker_frame, detections);

                if (face_detector) {
                    face_policy.annotate(*ctx, tracks, *face_detector);
                }
                ctx->face_detection_count = 0;
                for (const auto& track : tracks) {
                    if (track.face) ++ctx->face_detection_count;
                }
            }

            if (args.no_gallery || args.is_face_only()) {
                for (auto& track : tracks) {
                    track.identity_key.clear();
                    track.identity_confidence = 0.0f;
                    track.privacy_action = "anonymize";
                    track.recognition_state = "no_gallery";
                }
            } else {
                RecognitionTask rec_task;
                rec_task.stream_id = args.stream_id;
                rec_task.frame_id = frame_id;
                rec_task.frame = ctx;
                rec_task.tracks = tracks;
                const RecognitionResult rec_result = recognizer->recognize(rec_task);

                IdentityTask identity_task;
                identity_task.stream_id = args.stream_id;
                identity_task.frame_id = frame_id;
                identity_task.frame = ctx;
                identity_task.tracks = rec_result.tracks;
                const IdentityResult identity_result = identity->decide(identity_task);
                tracks = identity_result.tracks;
            }
            ctx->tracked_boxes = tracks;

            anonymizer.apply(ctx->ui, tracks, 1.0f, 1.0f, 0.0f, 0.0f);

            std::vector<veilsight::render::RenderEvent> frame_events =
                rule_engine.process_frame(frame_id, ctx->pts_ns, tracks);
            for (const auto& event : frame_events) {
                events_out << veilsight::render::render_event_to_json(event) << "\n";
                recent_events.push_back(event);
            }
            if (recent_events.size() > 20) {
                recent_events.erase(recent_events.begin(), recent_events.end() - 20);
            }

            cv::Mat out = ctx->ui.clone();
            if (layers.contains("rules")) draw_rules(out, rules);
            if (layers.contains("tracks") && !args.is_face_only()) draw_tracks(out, tracks);
            if (layers.contains("faces")) draw_faces(out, tracks);
            if (layers.contains("directions")) draw_directions(out, tracks, previous_positions);
            if (layers.contains("events")) draw_events(out, recent_events);
            writer.write(out);
            const int64_t processed_frame = frame_id + 1;
            if (!args.preview_jpeg.empty() && (processed_frame == 1 || processed_frame % args.preview_every_frames == 0)) {
                if (!write_preview_jpeg(preview_jpeg_path, out)) {
                    std::cerr << "Failed to write preview JPEG: " << preview_jpeg_path << "\n";
                }
            }

            previous_positions.clear();
            for (const Box& track : tracks) {
                if (track.id >= 0) previous_positions[track.id] = foot_point(track);
            }
        } catch (const std::exception& e) {
            std::cerr << "Frame " << frame_id << " error: " << e.what() << "\n";
            writer.write(ctx->ui);
        }

        const int64_t processed_frame = frame_id + 1;
        if (processed_frame % args.preview_every_frames == 0 || processed_frame == 1) {
            std::cout << "progress frame=" << processed_frame;
            if (total_frames > 0) std::cout << " total=" << total_frames;
            if (!args.preview_jpeg.empty()) std::cout << " preview=" << preview_jpeg_path.string();
            std::cout << "\n" << std::flush;
        }
        ++frame_id;
    } while (cap.read(frame) && !frame.empty());

    writer.release();
    events_out.close();

    try {
        write_manifest(manifest_path, args, frame_id, effective_fps, source_fps, layers, events_path);
    } catch (const std::exception& e) {
        std::cerr << "Manifest error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Processed " << frame_id << " frames\n";
    std::cout << "Wrote " << output_path << "\n";
    std::cout << "Events " << events_path << "\n";
    std::cout << "Manifest " << manifest_path << "\n";
    return 0;
}
