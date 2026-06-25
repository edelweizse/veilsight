#include <recognizer/recognizer.hpp>

#include <common/ncnn_options.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <opencv2/core.hpp>
#include <sqlite3.h>

namespace {
    int g_failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++g_failures;
            std::cerr << "[FAIL] " << message << "\n";
        }
    }

    std::filesystem::path find_repo_file(const std::filesystem::path& relative) {
        namespace fs = std::filesystem;
        fs::path dir = fs::current_path();
        for (int i = 0; i < 8; ++i) {
            const fs::path candidate = dir / relative;
            if (fs::exists(candidate)) return candidate;
            if (!dir.has_parent_path()) break;
            dir = dir.parent_path();
        }

        const fs::path source_relative = fs::path(__FILE__).parent_path().parent_path() / relative;
        if (fs::exists(source_relative)) return source_relative;
        return relative;
    }

    std::filesystem::path temp_db_path(const std::string& prefix) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() / (prefix + "_" + std::to_string(stamp) + ".sqlite3");
    }

    veilsight::RecognizerModuleConfig mobilefacenet_cfg() {
        veilsight::RecognizerModuleConfig cfg;
        cfg.type = "mobilefacenet";
        cfg.param_path = find_repo_file("models/face_embeddings/mobilefacenet/mobilefacenets.param").string();
        cfg.bin_path = find_repo_file("models/face_embeddings/mobilefacenet/mobilefacenets.bin").string();
        cfg.ncnn_threads = 1;
        cfg.unknown_threshold = 0.45f;
        return cfg;
    }

    veilsight::FaceObservation good_face(float score = 0.95f) {
        veilsight::FaceObservation face;
        face.bbox = veilsight::RectF{30.0f, 35.0f, 90.0f, 95.0f};
        face.score = score;
        face.landmark_count = 5;
        face.fresh = true;
        face.frame_id = 1;
        face.landmarks[0] = veilsight::PointF{52.0f, 62.0f};
        face.landmarks[1] = veilsight::PointF{92.0f, 62.0f};
        face.landmarks[2] = veilsight::PointF{72.0f, 82.0f};
        face.landmarks[3] = veilsight::PointF{57.0f, 108.0f};
        face.landmarks[4] = veilsight::PointF{87.0f, 108.0f};
        return face;
    }

    veilsight::Box track_with_face(int id, veilsight::FaceObservation face) {
        veilsight::Box track;
        track.id = id;
        track.x = 20.0f;
        track.y = 20.0f;
        track.w = 120.0f;
        track.h = 150.0f;
        track.score = 0.9f;
        track.privacy_action = "anonymize";
        track.face = face;
        return track;
    }

    veilsight::FramePtr frame(int64_t frame_id) {
        auto out = std::make_shared<veilsight::FrameCtx>();
        out->stream_id = "cam0";
        out->frame_id = frame_id;
        out->inf_w = 160;
        out->inf_h = 160;
        out->ui_w = 160;
        out->ui_h = 160;
        out->inf = cv::Mat(160, 160, CV_8UC3);
        out->ui = cv::Mat(160, 160, CV_8UC3);
        for (int y = 0; y < out->inf.rows; ++y) {
            for (int x = 0; x < out->inf.cols; ++x) {
                out->inf.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<unsigned char>((x * 3 + y) % 256),
                    static_cast<unsigned char>((x + y * 5) % 256),
                    static_cast<unsigned char>((x * 7 + y * 2) % 256));
            }
        }
        out->ui = out->inf.clone();
        return out;
    }

    std::vector<float> l2_normalize(std::vector<float> values) {
        double sum = 0.0;
        for (const float value : values) sum += static_cast<double>(value) * value;
        const float inv = 1.0f / static_cast<float>(std::sqrt(sum));
        for (float& value : values) value *= inv;
        return values;
    }

    std::vector<float> compute_embedding_for_test(const veilsight::RecognizerModuleConfig& cfg,
                                                  const veilsight::FramePtr& frame_ptr,
                                                  const veilsight::FaceObservation& face) {
        ncnn::Net net;
        veilsight::configure_ncnn_net(net, 1);
        if (net.load_param(cfg.param_path.c_str()) != 0) {
            throw std::runtime_error("failed to load test MobileFaceNet param");
        }
        if (net.load_model(cfg.bin_path.c_str()) != 0) {
            throw std::runtime_error("failed to load test MobileFaceNet weights");
        }

        std::vector<float> src(10);
        for (int i = 0; i < 5; ++i) {
            src[static_cast<size_t>(i * 2)] = face.landmarks[static_cast<size_t>(i)].x;
            src[static_cast<size_t>(i * 2 + 1)] = face.landmarks[static_cast<size_t>(i)].y;
        }
        const std::vector<float> canonical = {
            38.2946f, 51.6963f,
            73.5318f, 51.5014f,
            56.0252f, 71.7366f,
            41.5493f, 92.3655f,
            70.7299f, 92.2041f,
        };

        float tm_src_to_dst[6] = {};
        float tm_dst_to_src[6] = {};
        ncnn::get_affine_transform(src.data(), canonical.data(), 5, tm_src_to_dst);
        ncnn::invert_affine_transform(tm_src_to_dst, tm_dst_to_src);

        std::vector<unsigned char> aligned(112u * 112u * 3u);
        const cv::Mat& bgr = frame_ptr->inf;
        ncnn::warpaffine_bilinear_c3(
            bgr.data,
            bgr.cols,
            bgr.rows,
            static_cast<int>(bgr.step),
            aligned.data(),
            112,
            112,
            112 * 3,
            tm_dst_to_src);

        ncnn::Mat in = ncnn::Mat::from_pixels(aligned.data(), ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
        ncnn::Extractor ex = net.create_extractor();
        if (ex.input(cfg.input_blob.c_str(), in) != 0) {
            throw std::runtime_error("failed to feed test MobileFaceNet input");
        }
        ncnn::Mat out;
        if (ex.extract(cfg.output_blob.c_str(), out) != 0) {
            throw std::runtime_error("failed to extract test MobileFaceNet output");
        }
        if (static_cast<int>(out.total()) != 128) {
            throw std::runtime_error("test MobileFaceNet output dimension mismatch");
        }
        const float* data = static_cast<const float*>(out.data);
        return l2_normalize(std::vector<float>(data, data + 128));
    }

    void exec_sql(sqlite3* db, const char* sql) {
        char* err = nullptr;
        if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
            std::string message = err ? err : "unknown sqlite error";
            sqlite3_free(err);
            throw std::runtime_error(message);
        }
    }

    void create_gallery_db(const std::filesystem::path& path,
                           const std::vector<std::pair<std::string, std::vector<float>>>& embeddings,
                           int dim = 128) {
        sqlite3* db = nullptr;
        if (sqlite3_open(path.string().c_str(), &db) != SQLITE_OK) {
            throw std::runtime_error("failed to create test gallery DB");
        }

        try {
            exec_sql(db,
                     "CREATE TABLE identities ("
                     "identity_key TEXT PRIMARY KEY,"
                     "display_name TEXT,"
                     "active INTEGER NOT NULL DEFAULT 1);"
                     "CREATE TABLE face_embeddings ("
                     "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                     "identity_key TEXT NOT NULL,"
                     "model TEXT NOT NULL DEFAULT 'mobilefacenet',"
                     "dim INTEGER NOT NULL DEFAULT 128,"
                     "embedding BLOB NOT NULL,"
                     "active INTEGER NOT NULL DEFAULT 1,"
                     "FOREIGN KEY(identity_key) REFERENCES identities(identity_key));");

            sqlite3_stmt* identity_stmt = nullptr;
            sqlite3_stmt* embedding_stmt = nullptr;
            if (sqlite3_prepare_v2(db,
                                   "INSERT OR IGNORE INTO identities(identity_key, display_name, active) VALUES (?, ?, 1)",
                                   -1,
                                   &identity_stmt,
                                   nullptr) != SQLITE_OK) {
                throw std::runtime_error("failed to prepare identity insert");
            }
            if (sqlite3_prepare_v2(db,
                                   "INSERT INTO face_embeddings(identity_key, model, dim, embedding, active) "
                                   "VALUES (?, 'mobilefacenet', ?, ?, 1)",
                                   -1,
                                   &embedding_stmt,
                                   nullptr) != SQLITE_OK) {
                sqlite3_finalize(identity_stmt);
                throw std::runtime_error("failed to prepare embedding insert");
            }

            for (const auto& item : embeddings) {
                sqlite3_bind_text(identity_stmt, 1, item.first.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_text(identity_stmt, 2, item.first.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(identity_stmt) != SQLITE_DONE) {
                    sqlite3_finalize(identity_stmt);
                    sqlite3_finalize(embedding_stmt);
                    throw std::runtime_error("failed to insert identity");
                }
                sqlite3_reset(identity_stmt);
                sqlite3_clear_bindings(identity_stmt);

                sqlite3_bind_text(embedding_stmt, 1, item.first.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_int(embedding_stmt, 2, dim);
                sqlite3_bind_blob(
                    embedding_stmt,
                    3,
                    item.second.data(),
                    static_cast<int>(item.second.size() * sizeof(float)),
                    SQLITE_TRANSIENT);
                if (sqlite3_step(embedding_stmt) != SQLITE_DONE) {
                    sqlite3_finalize(identity_stmt);
                    sqlite3_finalize(embedding_stmt);
                    throw std::runtime_error("failed to insert embedding");
                }
                sqlite3_reset(embedding_stmt);
                sqlite3_clear_bindings(embedding_stmt);
            }

            sqlite3_finalize(identity_stmt);
            sqlite3_finalize(embedding_stmt);
            sqlite3_close(db);
        } catch (...) {
            sqlite3_close(db);
            throw;
        }
    }

    void test_mobilefacenet_factory_loads_empty_gallery_and_caches_unknown() {
        auto cfg = mobilefacenet_cfg();
        cfg.gallery_path.clear();
        const auto factory = veilsight::create_recognizer_factory(cfg);
        check(factory->backend_threads() == 1, "mobilefacenet backend_threads should report ncnn_threads");

        auto recognizer = factory->create();
        veilsight::RecognitionTask task;
        task.stream_id = "cam0";
        task.frame_id = 1;
        task.frame = frame(1);
        task.tracks = {track_with_face(7, good_face())};
        const auto result = recognizer->recognize(task);

        check(result.tracks.size() == 1, "mobilefacenet should preserve track count");
        check(result.tracks[0].id == 7, "mobilefacenet should preserve tracker id");
        check(result.tracks[0].identity_key.empty(), "empty gallery should produce unknown identity");
        check(result.tracks[0].identity_confidence == 0.0f, "empty gallery unknown confidence should be zero");
        check(result.tracks[0].privacy_action == "anonymize", "empty gallery should anonymize");
        check(task.frame->tracked_boxes.size() == 1 &&
                  task.frame->tracked_boxes[0].privacy_action == "anonymize",
              "mobilefacenet should update frame tracks");
    }

    void test_gallery_db_loads_multiple_embeddings_and_rejects_invalid_rows() {
        auto cfg = mobilefacenet_cfg();
        const auto valid_db = temp_db_path("veilsight_gallery_valid");
        std::vector<float> first(128, 0.0f);
        std::vector<float> second(128, 0.0f);
        first[0] = 1.0f;
        second[1] = 1.0f;
        create_gallery_db(valid_db, {{"alice", first}, {"alice", second}});
        cfg.gallery_path = valid_db.string();
        try {
            const auto factory = veilsight::create_recognizer_factory(cfg);
            check(factory != nullptr, "valid gallery with multiple embeddings should load");
        } catch (const std::exception& e) {
            check(false, std::string("valid gallery should not throw: ") + e.what());
        }
        std::filesystem::remove(valid_db);

        const auto invalid_db = temp_db_path("veilsight_gallery_invalid");
        std::vector<float> wrong_dim(127, 0.0f);
        wrong_dim[0] = 1.0f;
        create_gallery_db(invalid_db, {{"bob", wrong_dim}}, 127);
        cfg.gallery_path = invalid_db.string();
        bool threw = false;
        try {
            (void)veilsight::create_recognizer_factory(cfg);
        } catch (...) {
            threw = true;
        }
        check(threw, "wrong-dimension gallery embedding should fail startup");
        std::filesystem::remove(invalid_db);

        cfg.gallery_path = (std::filesystem::temp_directory_path() / "missing_veilsight_gallery.sqlite3").string();
        threw = false;
        try {
            (void)veilsight::create_recognizer_factory(cfg);
        } catch (...) {
            threw = true;
        }
        check(threw, "missing configured gallery DB should fail startup");
    }

    void test_mobilefacenet_gallery_self_match_allows() {
        auto cfg = mobilefacenet_cfg();
        const auto f = frame(2);
        const auto face = good_face();
        const auto embedding = compute_embedding_for_test(cfg, f, face);

        const auto db_path = temp_db_path("veilsight_gallery_self_match");
        create_gallery_db(db_path, {{"alice", embedding}});
        cfg.gallery_path = db_path.string();

        auto recognizer = veilsight::create_recognizer(cfg);
        veilsight::RecognitionTask task;
        task.stream_id = "cam0";
        task.frame_id = 2;
        task.frame = f;
        task.tracks = {track_with_face(22, face)};
        const auto result = recognizer->recognize(task);

        check(result.tracks.size() == 1, "self-match should preserve track count");
        check(result.tracks[0].identity_key == "alice", "self-match should assign gallery identity");
        check(result.tracks[0].identity_confidence > 0.99f, "self-match confidence should be near cosine 1");
        check(result.tracks[0].privacy_action == "allow", "self-match should allow known identity");
        std::filesystem::remove(db_path);
    }

    void test_mobilefacenet_face_only_box_can_match_gallery() {
        auto cfg = mobilefacenet_cfg();
        const auto f = frame(5);
        const auto face = good_face();
        const auto embedding = compute_embedding_for_test(cfg, f, face);

        const auto db_path = temp_db_path("veilsight_gallery_face_only");
        create_gallery_db(db_path, {{"alice", embedding}});
        cfg.gallery_path = db_path.string();

        auto recognizer = veilsight::create_recognizer(cfg);
        veilsight::RecognitionTask task;
        task.stream_id = "cam0";
        task.frame_id = 5;
        task.frame = f;
        task.tracks = {track_with_face(-1, face)};
        const auto result = recognizer->recognize(task);

        check(result.tracks.size() == 1, "face-only self-match should preserve track count");
        check(result.tracks[0].id == -1, "face-only self-match should preserve negative diagnostic id");
        check(result.tracks[0].identity_key == "alice", "face-only self-match should assign gallery identity");
        check(result.tracks[0].privacy_action == "allow", "face-only self-match should allow known identity");
        std::filesystem::remove(db_path);
    }

    void test_duplicate_face_only_detections_keep_best_match() {
        auto cfg = mobilefacenet_cfg();
        const auto f = frame(6);
        const auto best_face = good_face(0.95f);
        auto duplicate_face = best_face;
        duplicate_face.score = 0.90f;
        const auto embedding = compute_embedding_for_test(cfg, f, best_face);

        const auto db_path = temp_db_path("veilsight_gallery_duplicate_face_only");
        create_gallery_db(db_path, {{"alice", embedding}});
        cfg.gallery_path = db_path.string();

        auto recognizer = veilsight::create_recognizer(cfg);
        veilsight::RecognitionTask task;
        task.stream_id = "cam0";
        task.frame_id = 6;
        task.frame = f;
        task.tracks = {
            track_with_face(-1, best_face),
            track_with_face(-2, duplicate_face),
        };
        const auto result = recognizer->recognize(task);

        check(result.tracks.size() == 2, "duplicate face-only recognition should preserve tracks");
        check(result.tracks[0].identity_key == "alice",
              "best overlapping face-only detection should still match gallery");
        check(result.tracks[0].privacy_action == "allow",
              "best overlapping face-only detection should allow known identity");
        check(result.tracks[1].identity_key.empty() && result.tracks[1].recognition_state == "skipped",
              "lower-score duplicate face-only detection should be skipped");
        std::filesystem::remove(db_path);
    }

    void test_low_quality_attempt_does_not_cache_unknown() {
        auto cfg = mobilefacenet_cfg();
        const auto f = frame(3);
        const auto face = good_face();
        const auto embedding = compute_embedding_for_test(cfg, f, face);

        const auto db_path = temp_db_path("veilsight_gallery_quality_retry");
        create_gallery_db(db_path, {{"alice", embedding}});
        cfg.gallery_path = db_path.string();

        auto recognizer = veilsight::create_recognizer(cfg);

        auto low_quality = face;
        low_quality.score = 0.1f;
        veilsight::RecognitionTask first;
        first.stream_id = "cam0";
        first.frame_id = 3;
        first.frame = f;
        first.tracks = {track_with_face(33, low_quality)};
        const auto first_result = recognizer->recognize(first);
        check(first_result.tracks[0].identity_key.empty() &&
                  first_result.tracks[0].privacy_action == "anonymize",
              "low-quality face should not produce a known decision");

        veilsight::RecognitionTask second;
        second.stream_id = "cam0";
        second.frame_id = 4;
        second.frame = f;
        second.tracks = {track_with_face(33, face)};
        const auto second_result = recognizer->recognize(second);
        check(second_result.tracks[0].identity_key == "alice" &&
                  second_result.tracks[0].privacy_action == "allow",
              "later high-quality face should retry after low-quality non-decision");
        std::filesystem::remove(db_path);
    }
}

int main() {
    test_mobilefacenet_factory_loads_empty_gallery_and_caches_unknown();
    test_gallery_db_loads_multiple_embeddings_and_rejects_invalid_rows();
    test_mobilefacenet_gallery_self_match_allows();
    test_mobilefacenet_face_only_box_can_match_gallery();
    test_duplicate_face_only_detections_keep_best_match();
    test_low_quality_attempt_does_not_cache_unknown();

    if (g_failures != 0) {
        std::cerr << "[FAIL] total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "[OK] all recognizer tests passed\n";
    return 0;
}
