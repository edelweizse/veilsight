#include <recognizer/recognizer.hpp>

#include <face_detector/face_detector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ncnn/allocator.h>
#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <opencv2/core.hpp>
#include <sqlite3.h>

namespace veilsight {
    namespace {
        constexpr float kDuplicateFaceIou = 0.45f;
        constexpr float kPi = 3.14159265358979323846f;

        struct GalleryEntry {
            std::string identity_key;
            std::vector<float> embedding;
        };

        struct Gallery {
            std::vector<GalleryEntry> entries;
        };

        struct SharedGallery {
            mutable std::mutex mutex;
            std::shared_ptr<const Gallery> gallery = std::make_shared<Gallery>();
        };

        enum class TrackRecognitionState {
            InProgress,
            DecidedKnown,
            DecidedUnknown,
        };

        struct TrackDecision {
            TrackRecognitionState state = TrackRecognitionState::InProgress;
            std::string identity_key;
            float identity_confidence = 0.0f;
            std::string privacy_action = "anonymize";
            std::string recognition_state = "pending";
            int64_t last_seen_frame = 0;
        };

        struct StreamTrackState {
            std::unordered_map<int, TrackDecision> tracks;
        };

        struct SharedTrackState {
            std::mutex mutex;
            std::unordered_map<std::string, StreamTrackState> streams;
        };

        struct MatchResult {
            std::string identity_key;
            float score = 0.0f;
        };

        float area_of(const RectF& b) {
            return std::max(0.0f, b.w) * std::max(0.0f, b.h);
        }

        float iou_of(const RectF& a, const RectF& b) {
            const float ax2 = a.x + a.w;
            const float ay2 = a.y + a.h;
            const float bx2 = b.x + b.w;
            const float by2 = b.y + b.h;

            const float xx1 = std::max(a.x, b.x);
            const float yy1 = std::max(a.y, b.y);
            const float xx2 = std::min(ax2, bx2);
            const float yy2 = std::min(ay2, by2);

            const float iw = std::max(0.0f, xx2 - xx1);
            const float ih = std::max(0.0f, yy2 - yy1);
            const float inter = iw * ih;
            if (inter <= 0.0f) return 0.0f;

            const float uni = area_of(a) + area_of(b) - inter;
            if (uni <= 0.0f) return 0.0f;
            return inter / uni;
        }

        float distance(const PointF& a, const PointF& b) {
            const float dx = a.x - b.x;
            const float dy = a.y - b.y;
            return std::sqrt(dx * dx + dy * dy);
        }

        bool normalize_l2(std::vector<float>& values) {
            double sum = 0.0;
            for (const float v : values) {
                sum += static_cast<double>(v) * static_cast<double>(v);
            }
            if (sum <= 0.0) return false;

            const float inv_norm = 1.0f / static_cast<float>(std::sqrt(sum));
            for (float& v : values) v *= inv_norm;
            return true;
        }

        void apply_no_decision(Box& track) {
            track.identity_key.clear();
            track.identity_confidence = 0.0f;
            track.privacy_action = "anonymize";
            track.recognition_state = "skipped";
        }

        void apply_decision(Box& track, const TrackDecision& decision) {
            track.identity_key = decision.identity_key;
            track.identity_confidence = decision.identity_confidence;
            track.privacy_action = decision.privacy_action;
            track.recognition_state = decision.recognition_state;
        }

        std::string resolve_path_or_throw(const std::string& p) {
            namespace fs = std::filesystem;
            if (fs::exists(fs::path(p))) return p;
            const fs::path candidates[] = {
                fs::path("../") / p,
                fs::path("../../") / p,
                fs::path("../../../") / p,
            };
            for (const auto& candidate : candidates) {
                if (fs::exists(candidate)) return candidate.string();
            }
            throw std::runtime_error("Path not found: " + p);
        }

        class SqliteDb {
        public:
            explicit SqliteDb(const std::string& path) {
                if (sqlite3_open_v2(resolve_path_or_throw(path).c_str(), &db_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
                    std::string message = "failed to open gallery DB: " + path;
                    if (db_) message += ": " + std::string(sqlite3_errmsg(db_));
                    close();
                    throw std::runtime_error(message);
                }
            }

            SqliteDb(const SqliteDb&) = delete;
            SqliteDb& operator=(const SqliteDb&) = delete;

            ~SqliteDb() {
                close();
            }

            sqlite3* get() const {
                return db_;
            }

        private:
            void close() {
                if (db_) {
                    sqlite3_close(db_);
                    db_ = nullptr;
                }
            }

            sqlite3* db_ = nullptr;
        };

        class SqliteStmt {
        public:
            SqliteStmt(sqlite3* db, const char* sql) : db_(db) {
                if (sqlite3_prepare_v2(db_, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
                    throw std::runtime_error("failed to prepare gallery query: " + std::string(sqlite3_errmsg(db_)));
                }
            }

            SqliteStmt(const SqliteStmt&) = delete;
            SqliteStmt& operator=(const SqliteStmt&) = delete;

            ~SqliteStmt() {
                if (stmt_) sqlite3_finalize(stmt_);
            }

            sqlite3_stmt* get() const {
                return stmt_;
            }

        private:
            sqlite3* db_ = nullptr;
            sqlite3_stmt* stmt_ = nullptr;
        };

        std::shared_ptr<const Gallery> load_gallery(const RecognizerModuleConfig& cfg) {
            auto gallery = std::make_shared<Gallery>();
            if (cfg.gallery_path.empty()) return gallery;

            SqliteDb db(cfg.gallery_path);
            SqliteStmt stmt(
                db.get(),
                "SELECT fe.identity_key, fe.dim, fe.embedding "
                "FROM face_embeddings fe "
                "JOIN identities i ON i.identity_key = fe.identity_key "
                "WHERE fe.active = 1 AND i.active = 1 AND fe.model = 'mobilefacenet' "
                "ORDER BY fe.identity_key, fe.id");

            while (true) {
                const int rc = sqlite3_step(stmt.get());
                if (rc == SQLITE_DONE) break;
                if (rc != SQLITE_ROW) {
                    throw std::runtime_error("failed to read gallery rows: " + std::string(sqlite3_errmsg(db.get())));
                }

                const auto* key_text = sqlite3_column_text(stmt.get(), 0);
                const int dim = sqlite3_column_int(stmt.get(), 1);
                const void* blob = sqlite3_column_blob(stmt.get(), 2);
                const int bytes = sqlite3_column_bytes(stmt.get(), 2);

                if (!key_text || !blob) {
                    throw std::runtime_error("gallery contains null identity_key or embedding");
                }
                if (dim != cfg.embedding_dim) {
                    throw std::runtime_error("gallery embedding has dim " + std::to_string(dim) +
                                             ", expected " + std::to_string(cfg.embedding_dim));
                }
                const int expected_bytes = cfg.embedding_dim * static_cast<int>(sizeof(float));
                if (bytes != expected_bytes) {
                    throw std::runtime_error("gallery embedding has " + std::to_string(bytes) +
                                             " bytes, expected " + std::to_string(expected_bytes));
                }

                GalleryEntry entry;
                entry.identity_key = reinterpret_cast<const char*>(key_text);
                entry.embedding.resize(static_cast<size_t>(cfg.embedding_dim));
                std::memcpy(entry.embedding.data(), blob, static_cast<size_t>(expected_bytes));
                if (!normalize_l2(entry.embedding)) {
                    throw std::runtime_error("gallery embedding for " + entry.identity_key + " has zero norm");
                }
                gallery->entries.push_back(std::move(entry));
            }

            return gallery;
        }

        bool passes_quality_gates(const Box& track,
                                  size_t track_index,
                                  const std::vector<Box>& tracks,
                                  const RecognizerModuleConfig& cfg) {
            if (!track.face) return false;
            const FaceObservation& face = *track.face;
            if (!face.fresh) return false;
            if (face.landmark_count != 5) return false;
            if (face.score < cfg.min_face_score) return false;
            if (face.bbox.w < cfg.min_face_size_px || face.bbox.h < cfg.min_face_size_px) return false;

            const PointF& left_eye = face.landmarks[0];
            const PointF& right_eye = face.landmarks[1];
            const PointF& nose = face.landmarks[2];
            const PointF& left_mouth = face.landmarks[3];
            const PointF& right_mouth = face.landmarks[4];

            const float inter_eye = distance(left_eye, right_eye);
            if (inter_eye < cfg.min_inter_eye_px) return false;

            const float roll_deg = std::abs(std::atan2(right_eye.y - left_eye.y,
                                                       right_eye.x - left_eye.x)) *
                                   180.0f / kPi;
            if (roll_deg > cfg.max_roll_deg) return false;

            const PointF eye_mid{
                (left_eye.x + right_eye.x) * 0.5f,
                (left_eye.y + right_eye.y) * 0.5f,
            };
            const PointF mouth_mid{
                (left_mouth.x + right_mouth.x) * 0.5f,
                (left_mouth.y + right_mouth.y) * 0.5f,
            };

            if (!(nose.y > eye_mid.y && nose.y < mouth_mid.y)) return false;
            if (!(mouth_mid.y > nose.y)) return false;
            if (std::abs(nose.x - eye_mid.x) / inter_eye > cfg.max_yaw_offset_ratio) return false;

            for (size_t i = 0; i < tracks.size(); ++i) {
                if (i == track_index) continue;
                if (!tracks[i].face || !tracks[i].face->fresh) continue;
                if (iou_of(face.bbox, tracks[i].face->bbox) <= kDuplicateFaceIou) continue;
                const FaceObservation& other = *tracks[i].face;
                if (other.score > face.score || (other.score == face.score && i < track_index)) return false;
            }

            return true;
        }

        MatchResult best_gallery_match(const std::vector<float>& embedding, const Gallery& gallery) {
            MatchResult out;
            out.score = gallery.entries.empty() ? 0.0f : -std::numeric_limits<float>::infinity();
            for (const auto& entry : gallery.entries) {
                float score = 0.0f;
                for (size_t i = 0; i < embedding.size(); ++i) {
                    score += embedding[i] * entry.embedding[i];
                }
                if (score > out.score) {
                    out.score = score;
                    out.identity_key = entry.identity_key;
                }
            }
            if (gallery.entries.empty()) out.score = 0.0f;
            return out;
        }

        std::vector<float> extract_mobilefacenet_embedding(ncnn::Net& net,
                                                           ncnn::PoolAllocator& workspace_pool_allocator,
                                                           const RecognizerModuleConfig& cfg,
                                                           const cv::Mat& bgr,
                                                           const FaceObservation& face) {
            if (bgr.empty()) {
                throw std::runtime_error("recognition image is empty");
            }
            if (bgr.type() != CV_8UC3) {
                throw std::runtime_error("recognition image must be CV_8UC3");
            }

            std::array<float, 10> src{};
            for (int i = 0; i < 5; ++i) {
                src[static_cast<size_t>(i * 2)] = face.landmarks[static_cast<size_t>(i)].x;
                src[static_cast<size_t>(i * 2 + 1)] = face.landmarks[static_cast<size_t>(i)].y;
            }

            static constexpr std::array<float, 10> canonical = {
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

            std::vector<unsigned char> aligned(
                static_cast<size_t>(cfg.input_w) * static_cast<size_t>(cfg.input_h) * 3u);
            ncnn::warpaffine_bilinear_c3(
                bgr.data,
                bgr.cols,
                bgr.rows,
                static_cast<int>(bgr.step),
                aligned.data(),
                cfg.input_w,
                cfg.input_h,
                cfg.input_w * 3,
                tm_dst_to_src);

            ncnn::Mat in = ncnn::Mat::from_pixels(
                aligned.data(),
                ncnn::Mat::PIXEL_BGR2RGB,
                cfg.input_w,
                cfg.input_h);

            ncnn::Extractor ex = net.create_extractor();
            ex.set_light_mode(true);

            thread_local ncnn::UnlockedPoolAllocator blob_pool_allocator;
            thread_local bool blob_pool_initialized = false;
            if (!blob_pool_initialized) {
                blob_pool_allocator.set_size_compare_ratio(0.0f);
                blob_pool_initialized = true;
            }
            ex.set_blob_allocator(&blob_pool_allocator);
            ex.set_workspace_allocator(&workspace_pool_allocator);

            if (ex.input(cfg.input_blob.c_str(), in) != 0) {
                throw std::runtime_error("MobileFaceNet input failed for blob: " + cfg.input_blob);
            }

            ncnn::Mat fc1;
            if (ex.extract(cfg.output_blob.c_str(), fc1) != 0) {
                throw std::runtime_error("MobileFaceNet output failed for blob: " + cfg.output_blob);
            }
            if (static_cast<int>(fc1.total()) != cfg.embedding_dim) {
                throw std::runtime_error("MobileFaceNet output dimension mismatch");
            }

            const float* data = static_cast<const float*>(fc1.data);
            if (!data) throw std::runtime_error("MobileFaceNet output is empty");

            std::vector<float> embedding(static_cast<size_t>(cfg.embedding_dim));
            std::copy(data, data + cfg.embedding_dim, embedding.begin());
            if (!normalize_l2(embedding)) {
                throw std::runtime_error("MobileFaceNet output has zero norm");
            }
            return embedding;
        }

        std::vector<std::string> enrollment_reject_reasons(const FaceObservation& face,
                                                           const RecognizerModuleConfig& cfg) {
            std::vector<std::string> reasons;
            if (face.landmark_count != 5) reasons.emplace_back("missing_landmarks");
            if (face.score < cfg.min_face_score) reasons.emplace_back("low_face_score");
            if (face.bbox.w < cfg.min_face_size_px || face.bbox.h < cfg.min_face_size_px) {
                reasons.emplace_back("face_too_small");
            }
            if (face.landmark_count == 5) {
                const PointF& left_eye = face.landmarks[0];
                const PointF& right_eye = face.landmarks[1];
                const PointF& nose = face.landmarks[2];
                const PointF& left_mouth = face.landmarks[3];
                const PointF& right_mouth = face.landmarks[4];
                const float inter_eye = distance(left_eye, right_eye);
                if (inter_eye < cfg.min_inter_eye_px) reasons.emplace_back("inter_eye_too_small");
                const float roll_deg = std::abs(std::atan2(right_eye.y - left_eye.y,
                                                           right_eye.x - left_eye.x)) *
                                       180.0f / kPi;
                if (roll_deg > cfg.max_roll_deg) reasons.emplace_back("roll_too_large");
                const PointF eye_mid{
                    (left_eye.x + right_eye.x) * 0.5f,
                    (left_eye.y + right_eye.y) * 0.5f,
                };
                const PointF mouth_mid{
                    (left_mouth.x + right_mouth.x) * 0.5f,
                    (left_mouth.y + right_mouth.y) * 0.5f,
                };
                if (!(nose.y > eye_mid.y && nose.y < mouth_mid.y) || !(mouth_mid.y > nose.y)) {
                    reasons.emplace_back("invalid_landmark_geometry");
                }
                if (inter_eye > 0.0f && std::abs(nose.x - eye_mid.x) / inter_eye > cfg.max_yaw_offset_ratio) {
                    reasons.emplace_back("yaw_too_large");
                }
            }
            return reasons;
        }

        FaceDetectorRunConfig enrollment_run_config(const FaceDetectorModuleConfig& cfg) {
            if (cfg.type == "yunet") {
                return FaceDetectorRunConfig{std::max(1, cfg.yunet.input_w), std::max(1, cfg.yunet.input_h)};
            }
            return FaceDetectorRunConfig{std::max(1, cfg.scrfd.input_w), std::max(1, cfg.scrfd.input_h)};
        }

        class MobileFaceNetRecognizer final : public IRecognizer {
        public:
            MobileFaceNetRecognizer(RecognizerModuleConfig cfg,
                                    std::shared_ptr<SharedGallery> gallery,
                                    std::shared_ptr<SharedTrackState> state)
                : cfg_(std::move(cfg)),
                  gallery_(std::move(gallery)),
                  state_(std::move(state)) {
                if (!gallery_) gallery_ = std::make_shared<SharedGallery>();
                if (!state_) state_ = std::make_shared<SharedTrackState>();

                net_.opt.use_vulkan_compute = false;
                net_.opt.num_threads = std::max(1, cfg_.ncnn_threads);
                workspace_pool_allocator_.set_size_compare_ratio(0.0f);

                const std::string param = resolve_path_or_throw(cfg_.param_path);
                const std::string bin = resolve_path_or_throw(cfg_.bin_path);
                if (net_.load_param(param.c_str()) != 0) {
                    throw std::runtime_error("Failed to load MobileFaceNet param: " + param);
                }
                if (net_.load_model(bin.c_str()) != 0) {
                    throw std::runtime_error("Failed to load MobileFaceNet weights: " + bin);
                }
            }

            RecognitionResult recognize(const RecognitionTask& task) override {
                RecognitionResult out;
                out.stream_id = task.stream_id;
                out.frame_id = task.frame_id;
                out.frame = task.frame;
                out.tracks = task.tracks;

                cleanup_cache(task.stream_id, task.frame_id);

                for (size_t i = 0; i < out.tracks.size(); ++i) {
                    Box& track = out.tracks[i];
                    apply_no_decision(track);
                    const bool cacheable_track = track.id >= 0;

                    if (cacheable_track && apply_cached_decision(task.stream_id, track.id, task.frame_id, track)) {
                        continue;
                    }

                    if (!passes_quality_gates(track, i, out.tracks, cfg_)) {
                        continue;
                    }

                    if (cacheable_track && !mark_in_progress(task.stream_id, track.id, task.frame_id, track)) {
                        continue;
                    }

                    try {
                        const std::vector<float> embedding = extract_embedding(task.frame, *track.face);
                        const auto gallery = current_gallery();
                        const MatchResult match = best_gallery_match(embedding, *gallery);

                        TrackDecision decision;
                        decision.last_seen_frame = task.frame_id;
                        if (!gallery->entries.empty() && match.score >= cfg_.unknown_threshold) {
                            decision.state = TrackRecognitionState::DecidedKnown;
                            decision.identity_key = match.identity_key;
                            decision.identity_confidence = match.score;
                            decision.privacy_action = "allow";
                            decision.recognition_state = "known";
                        } else {
                            decision.state = TrackRecognitionState::DecidedUnknown;
                            decision.identity_key.clear();
                            decision.identity_confidence = gallery->entries.empty() ? 0.0f : match.score;
                            decision.privacy_action = "anonymize";
                            decision.recognition_state = "unknown";
                        }
                        if (cacheable_track) {
                            finish_decision(task.stream_id, track.id, decision, track);
                        } else {
                            apply_decision(track, decision);
                        }
                    } catch (...) {
                        track.recognition_state = "failed";
                        if (cacheable_track) {
                            clear_in_progress(task.stream_id, track.id);
                        }
                    }
                }

                if (out.frame) {
                    out.frame->tracked_boxes = out.tracks;
                }
                return out;
            }

        private:
            std::shared_ptr<const Gallery> current_gallery() const {
                std::lock_guard lk(gallery_->mutex);
                return gallery_->gallery ? gallery_->gallery : std::make_shared<Gallery>();
            }

            void cleanup_cache(const std::string& stream_id, int64_t frame_id) {
                std::lock_guard lk(state_->mutex);
                auto stream_it = state_->streams.find(stream_id);
                if (stream_it == state_->streams.end()) return;

                auto& tracks = stream_it->second.tracks;
                for (auto it = tracks.begin(); it != tracks.end();) {
                    const int64_t age = frame_id - it->second.last_seen_frame;
                    if (age > cfg_.cache_ttl_frames) {
                        it = tracks.erase(it);
                    } else {
                        ++it;
                    }
                }
                if (tracks.empty()) state_->streams.erase(stream_it);
            }

            bool apply_cached_decision(const std::string& stream_id,
                                       int track_id,
                                       int64_t frame_id,
                                       Box& track) {
                std::lock_guard lk(state_->mutex);
                auto stream_it = state_->streams.find(stream_id);
                if (stream_it == state_->streams.end()) return false;
                auto track_it = stream_it->second.tracks.find(track_id);
                if (track_it == stream_it->second.tracks.end()) return false;

                TrackDecision& decision = track_it->second;
                if (decision.state == TrackRecognitionState::InProgress) {
                    track.recognition_state = decision.recognition_state.empty() ? "pending" : decision.recognition_state;
                    return true;
                }

                decision.last_seen_frame = frame_id;
                apply_decision(track, decision);
                return true;
            }

            bool mark_in_progress(const std::string& stream_id,
                                  int track_id,
                                  int64_t frame_id,
                                  Box& track) {
                std::lock_guard lk(state_->mutex);
                auto& tracks = state_->streams[stream_id].tracks;
                auto it = tracks.find(track_id);
                if (it != tracks.end()) {
                    TrackDecision& decision = it->second;
                    if (decision.state == TrackRecognitionState::DecidedKnown ||
                        decision.state == TrackRecognitionState::DecidedUnknown) {
                        decision.last_seen_frame = frame_id;
                        apply_decision(track, decision);
                    } else {
                        track.recognition_state = decision.recognition_state.empty()
                        ? "pending" : decision.recognition_state;
                    }
                    return false;
                }

                TrackDecision decision;
                decision.state = TrackRecognitionState::InProgress;
                decision.recognition_state = "pending";
                decision.last_seen_frame = frame_id;
                tracks[track_id] = std::move(decision);
                track.recognition_state = "pending";
                return true;
            }

            void finish_decision(const std::string& stream_id,
                                 int track_id,
                                 const TrackDecision& decision,
                                 Box& track) {
                std::lock_guard lk(state_->mutex);
                state_->streams[stream_id].tracks[track_id] = decision;
                apply_decision(track, decision);
            }

            void clear_in_progress(const std::string& stream_id, int track_id) {
                std::lock_guard lk(state_->mutex);
                auto stream_it = state_->streams.find(stream_id);
                if (stream_it == state_->streams.end()) return;
                auto track_it = stream_it->second.tracks.find(track_id);
                if (track_it == stream_it->second.tracks.end()) return;
                if (track_it->second.state == TrackRecognitionState::InProgress) {
                    stream_it->second.tracks.erase(track_it);
                }
                if (stream_it->second.tracks.empty()) state_->streams.erase(stream_it);
            }

            std::vector<float> extract_embedding(const FramePtr& frame, const FaceObservation& face) {
                if (!frame || frame->inf.empty()) {
                    throw std::runtime_error("recognition frame has no inference image");
                }
                return extract_mobilefacenet_embedding(net_, workspace_pool_allocator_, cfg_, frame->inf, face);
            }

            RecognizerModuleConfig cfg_;
            std::shared_ptr<SharedGallery> gallery_;
            std::shared_ptr<SharedTrackState> state_;
            ncnn::Net net_;
            mutable ncnn::PoolAllocator workspace_pool_allocator_;
        };

        class MobileFaceNetRecognizerFactory final : public IRecognizerFactory {
        public:
            explicit MobileFaceNetRecognizerFactory(RecognizerModuleConfig cfg)
                : cfg_(std::move(cfg)),
                  gallery_(std::make_shared<SharedGallery>()),
                  state_(std::make_shared<SharedTrackState>()) {
                gallery_->gallery = load_gallery(cfg_);
            }

            std::unique_ptr<IRecognizer> create() const override {
                return std::make_unique<MobileFaceNetRecognizer>(cfg_, gallery_, state_);
            }

            int backend_threads() const override {
                return std::max(1, cfg_.ncnn_threads);
            }

            bool reload_gallery(std::string* error) override {
                try {
                    auto next = load_gallery(cfg_);
                    {
                        std::lock_guard lk(gallery_->mutex);
                        gallery_->gallery = std::move(next);
                    }
                    {
                        std::lock_guard lk(state_->mutex);
                        state_->streams.clear();
                    }
                    return true;
                } catch (const std::exception& e) {
                    if (error) *error = e.what();
                    return false;
                }
            }

        private:
            RecognizerModuleConfig cfg_;
            std::shared_ptr<SharedGallery> gallery_;
            std::shared_ptr<SharedTrackState> state_;
        };
    }

    EnrollmentAnalysisResult analyze_mobilefacenet_enrollment_image(
        const FaceDetectorModuleConfig& face_cfg,
        const RecognizerModuleConfig& recognizer_cfg,
        const cv::Mat& bgr) {
        EnrollmentAnalysisResult out;
        out.image_width = bgr.cols;
        out.image_height = bgr.rows;
        if (bgr.empty()) {
            out.ok = false;
            out.message = "image decode produced an empty frame";
            return out;
        }
        if (bgr.type() != CV_8UC3) {
            out.ok = false;
            out.message = "enrollment image must decode to CV_8UC3";
            return out;
        }

        try {
            auto detector = create_face_detector(face_cfg);
            FaceDetectorRunConfig run = enrollment_run_config(face_cfg);
            const auto faces = detector ? detector->detect_faces(bgr, run) : std::vector<FaceObservation>{};

            ncnn::Net net;
            net.opt.use_vulkan_compute = false;
            net.opt.num_threads = std::max(1, recognizer_cfg.ncnn_threads);
            ncnn::PoolAllocator workspace_pool_allocator;
            workspace_pool_allocator.set_size_compare_ratio(0.0f);
            const std::string param = resolve_path_or_throw(recognizer_cfg.param_path);
            const std::string bin = resolve_path_or_throw(recognizer_cfg.bin_path);
            if (net.load_param(param.c_str()) != 0) {
                throw std::runtime_error("Failed to load MobileFaceNet param: " + param);
            }
            if (net.load_model(bin.c_str()) != 0) {
                throw std::runtime_error("Failed to load MobileFaceNet weights: " + bin);
            }

            out.candidates.reserve(faces.size());
            for (const auto& face : faces) {
                EnrollmentAnalysisCandidate candidate;
                candidate.face = face;
                candidate.reject_reasons = enrollment_reject_reasons(face, recognizer_cfg);
                candidate.usable = candidate.reject_reasons.empty();
                if (face.landmark_count == 5) {
                    try {
                        candidate.embedding = extract_mobilefacenet_embedding(
                            net,
                            workspace_pool_allocator,
                            recognizer_cfg,
                            bgr,
                            face);
                    } catch (const std::exception& e) {
                        candidate.usable = false;
                        candidate.reject_reasons.emplace_back(std::string("embedding_failed: ") + e.what());
                    }
                }
                out.candidates.push_back(std::move(candidate));
            }
            out.ok = true;
            out.message = faces.empty() ? "no faces detected" : "ok";
            return out;
        } catch (const std::exception& e) {
            out.ok = false;
            out.message = e.what();
            return out;
        }
    }

    std::unique_ptr<IRecognizerFactory> create_mobilefacenet_recognizer_factory(
        const RecognizerModuleConfig& cfg) {
        return std::make_unique<MobileFaceNetRecognizerFactory>(cfg);
    }
}
