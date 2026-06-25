#include <common/config.hpp>
#include <face_detector/face_detector.hpp>
#include <recognizer/recognizer.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace veilsight;

namespace {
    struct Args {
        std::string config_path;
        std::string faces_dir;
        std::string output;
    };

    double embedding_norm(const std::vector<float>& emb) {
        double sum = 0.0;
        for (float v : emb) sum += static_cast<double>(v) * v;
        return std::sqrt(sum);
    }

    void usage(const char* prog) {
        std::cerr << "Usage: " << prog
                  << " --config <yaml> --faces-dir <dir> --output <json>\n";
    }

    Args parse_args(int argc, char** argv) {
        Args args;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* name) -> std::string {
                if (i + 1 >= argc)
                    throw std::runtime_error(std::string("missing value for ") + name);
                return argv[++i];
            };
            if (arg == "--config")
                args.config_path = require_value("--config");
            else if (arg == "--faces-dir")
                args.faces_dir = require_value("--faces-dir");
            else if (arg == "--output")
                args.output = require_value("--output");
            else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                std::exit(0);
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        }
        if (args.config_path.empty() || args.faces_dir.empty() || args.output.empty()) {
            throw std::runtime_error("config, faces-dir, and output are required");
        }
        return args;
    }

    int enroll_faces(const Args& args) {
        AppConfig cfg = load_config_yaml(args.config_path);

        const fs::path faces_dir(args.faces_dir);
        if (!fs::is_directory(faces_dir)) {
            std::cerr << "Not a directory: " << faces_dir << "\n";
            return 1;
        }

        auto& face_cfg = cfg.modules.face_detector;
        auto& recog_cfg = cfg.modules.recognizer;
        recog_cfg.gallery_path = "";

        std::ofstream out(args.output);
        if (!out) {
            std::cerr << "Cannot write output: " << args.output << "\n";
            return 1;
        }

        out << "[\n";

        std::vector<fs::path> pgm_files;
        for (const auto& entry : fs::directory_iterator(faces_dir)) {
            if (!entry.is_regular_file()) continue;
            fs::path path = entry.path();
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            if (ext == ".pgm") pgm_files.push_back(std::move(path));
        }
        std::sort(pgm_files.begin(), pgm_files.end());

        int entry_count = 0;
        for (const auto& path : pgm_files) {

            const std::string identity_key = path.parent_path().filename().string();

            cv::Mat gray = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
            if (gray.empty()) {
                std::cerr << "Skipping unreadable: " << path << "\n";
                continue;
            }

            cv::Mat bgr;
            cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);

            // Pad small face crops to give the detector enough context.
            // SCRFD requires the face to be a reasonable fraction of the input.
            const int kMinDim = 256;
            if (bgr.rows < kMinDim || bgr.cols < kMinDim) {
                int pad_h = std::max(0, (kMinDim - bgr.rows) / 2);
                int pad_w = std::max(0, (kMinDim - bgr.cols) / 2);
                int new_h = std::max(kMinDim, bgr.rows);
                int new_w = std::max(kMinDim, bgr.cols);
                cv::Mat padded(new_h, new_w, CV_8UC3, cv::Scalar(128, 128, 128));
                bgr.copyTo(padded(cv::Rect(pad_w, pad_h, bgr.cols, bgr.rows)));
                bgr = padded;
            }

            try {
                EnrollmentAnalysisResult result = analyze_mobilefacenet_enrollment_image(
                    face_cfg, recog_cfg, bgr);

                for (const auto& candidate : result.candidates) {
                    if (!candidate.usable || candidate.embedding.empty()) {
                        continue;
                    }
                    double norm = embedding_norm(candidate.embedding);
                    if (norm < 0.01 || norm > 100.0) {
                        std::cerr << "Embedding with suspicious norm " << norm
                                  << " for " << path.filename() << ", skipping\n";
                        continue;
                    }
                    std::string display_name = identity_key + "_" + path.stem().string();
                if (entry_count > 0) out << ",\n";
                out << "    {\n";
                out << "      \"identity_key\": \"" << identity_key << "\",\n";
                out << "      \"display_name\": \"" << display_name << "\",\n";
                out << "      \"embedding\": [";
                for (size_t i = 0; i < candidate.embedding.size(); ++i) {
                    if (i) out << ", ";
                    out << std::fixed << std::setprecision(6) << candidate.embedding[i];
                }
                out << "]\n";
                out << "    }";
                    ++entry_count;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << path.filename() << ": " << e.what() << "\n";
            }
        }

        out << (entry_count > 0 ? "\n" : "") << "]\n";
        out.close();

        std::cout << "Enrolled " << entry_count << " embeddings to " << args.output << "\n";
        return 0;
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
    return enroll_faces(args);
}
