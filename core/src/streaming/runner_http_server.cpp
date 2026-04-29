#include <streaming/runner_http_server.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>

#include <httplib.h>
#include <opencv2/imgcodecs.hpp>
#include <streaming/webrtc_publisher.hpp>

namespace veilsight {
    struct RunnerHTTPServer::Impl {
        httplib::Server svr;
    };

    RunnerHTTPServer::RunnerHTTPServer(std::string host, int port, StreamingConfig streaming)
        : impl_(new Impl()),
          host_(std::move(host)),
          port_(port),
          streaming_(std::move(streaming)) {}

    RunnerHTTPServer::~RunnerHTTPServer() {
        stop();
        delete impl_;
    }

    void RunnerHTTPServer::set_webrtc_publisher(WebRTCPublisher* publisher) {
        webrtc_ = publisher;
    }

    std::shared_ptr<RunnerHTTPServer::StreamState>
    RunnerHTTPServer::get_or_create_(const std::string& key) const {
        std::lock_guard lk(streams_mtx_);
        auto& p = streams_[key];
        if (!p) p = std::make_shared<StreamState>();
        return p;
    }

    std::shared_ptr<RunnerHTTPServer::StreamState>
    RunnerHTTPServer::get_(const std::string& key) const {
        std::lock_guard lk(streams_mtx_);
        auto it = streams_.find(key);
        if (it == streams_.end()) return nullptr;
        return it->second;
    }

    void RunnerHTTPServer::push_jpeg(const std::string& stream_key,
                                     std::shared_ptr<const std::vector<uint8_t>> jpeg) {
        auto st = get_or_create_(stream_key);
        {
            std::lock_guard lk(st->mtx);
            st->last_jpeg = std::move(jpeg);
            ++st->seq;
        }
        st->cv.notify_all();
    }

    void RunnerHTTPServer::push_jpeg(const std::string& stream_key,
                                     const cv::Mat& frame,
                                     int quality) {
        if (frame.empty() || frame.type() != CV_8UC3) return;
        std::vector<uint8_t> tmp;
        std::vector params = {cv::IMWRITE_JPEG_QUALITY, quality};
        if (!cv::imencode(".jpg", frame, tmp, params)) return;
        push_jpeg(stream_key, std::make_shared<std::vector<uint8_t>>(std::move(tmp)));
    }

    void RunnerHTTPServer::push_meta(const std::string& stream_key, std::string json) {
        auto st = get_or_create_(stream_key);
        std::lock_guard<std::mutex> lk(st->meta_mtx);
        st->last_meta = std::move(json);
    }

    void RunnerHTTPServer::push_metrics(std::string json) {
        std::lock_guard<std::mutex> lk(metrics_mtx_);
        last_metrics_json_ = std::move(json);
    }

    void RunnerHTTPServer::register_stream(const std::string& stream_key) {
        (void)get_or_create_(stream_key);
    }

    std::vector<std::string> RunnerHTTPServer::list_streams() const {
        std::lock_guard lk(streams_mtx_);
        std::vector<std::string> out;
        out.reserve(streams_.size());
        for (const auto& kv : streams_) out.push_back(kv.first);
        std::sort(out.begin(), out.end());
        return out;
    }

    std::string RunnerHTTPServer::cors_origin_(const std::string& request_origin) const {
        const auto& origins = streaming_.webrtc.cors_allowed_origins;
        if (!request_origin.empty() &&
            std::find(origins.begin(), origins.end(), request_origin) != origins.end()) {
            return request_origin;
        }
        return origins.empty() ? "*" : origins.front();
    }

    void RunnerHTTPServer::apply_cors_(const std::string& request_origin, httplib::Response& res) const {
        res.set_header("Access-Control-Allow-Origin", cors_origin_(request_origin));
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Expose-Headers", "Location");
        res.set_header("Vary", "Origin");
    }

    bool RunnerHTTPServer::start() {
        if (running_) return true;
        running_ = true;

        const auto origin = [](const httplib::Request& req) {
            auto it = req.headers.find("Origin");
            return it == req.headers.end() ? std::string{} : it->second;
        };

        impl_->svr.Options(R"(.*)", [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            res.status = 204;
        });

        impl_->svr.Get("/health", [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            res.set_content("{\"ok\":true}", "application/json");
        });

        impl_->svr.Get("/streams", [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            auto keys = list_streams();
            std::ostringstream oss;
            oss << "{\"streams\":[";
            for (size_t i = 0; i < keys.size(); ++i) {
                oss << "\"" << keys[i] << "\"";
                if (i + 1 < keys.size()) oss << ",";
            }
            oss << "],\"webrtc_available\":" << ((webrtc_ && webrtc_->available()) ? "true" : "false") << "}";
            res.set_content(oss.str(), "application/json");
            res.set_header("Cache-Control", "no-cache");
        });

        impl_->svr.Get("/metrics", [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            std::string payload;
            {
                std::lock_guard<std::mutex> lk(metrics_mtx_);
                payload = last_metrics_json_.empty() ? "{}" : last_metrics_json_;
            }
            res.set_content(payload, "application/json");
            res.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
            res.set_header("Pragma", "no-cache");
        });

        impl_->svr.Get(R"(/meta/(.+))", [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            if (req.matches.size() < 2) { res.status = 400; return; }
            const std::string key = req.matches[1];
            auto st = get_(key);
            if (!st) { res.status = 404; res.set_content("{}", "application/json"); return; }

            std::string json;
            {
                std::lock_guard<std::mutex> lk(st->meta_mtx);
                json = st->last_meta.empty() ? "{}" : st->last_meta;
            }
            res.set_content(json, "application/json");
            res.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
            res.set_header("Pragma", "no-cache");
        });

        impl_->svr.Get(R"(/snapshot/(.+))", [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            if (req.matches.size() < 2) { res.status = 400; return; }
            const std::string key = req.matches[1];
            auto st = get_(key);
            if (!st) { res.status = 404; return; }

            std::shared_ptr<const std::vector<uint8_t>> jpeg;
            {
                std::lock_guard<std::mutex> lk(st->mtx);
                jpeg = st->last_jpeg;
            }
            if (!jpeg || jpeg->empty()) { res.status = 204; return; }
            res.set_content(reinterpret_cast<const char*>(jpeg->data()), jpeg->size(), "image/jpeg");
            res.set_header("Cache-Control", "no-cache");
        });

        impl_->svr.Get(R"(/video/(.+))", [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            if (req.matches.size() < 2) { res.status = 400; return; }
            const std::string key = req.matches[1];
            auto st = get_(key);
            if (!st) { res.status = 404; return; }

            res.set_header("Cache-Control", "no-cache");
            res.set_header("Pragma", "no-cache");
            res.set_header("Connection", "close");
            res.set_header("X-Accel-Buffering", "no");

            const std::string boundary = "frame";
            res.set_content_provider(
                "multipart/x-mixed-replace; boundary=" + boundary,
                [this, st, boundary](size_t, httplib::DataSink& sink) {
                    uint64_t last_sent = 0;
                    while (running_) {
                        std::shared_ptr<const std::vector<uint8_t>> jpeg;
                        uint64_t seq_local = 0;
                        {
                            std::unique_lock lk(st->mtx);
                            st->cv.wait(lk, [&] { return st->seq != last_sent || !running_; });
                            if (!running_) break;
                            jpeg = st->last_jpeg;
                            seq_local = st->seq;
                        }
                        last_sent = seq_local;
                        if (!jpeg || jpeg->empty()) continue;
                        std::string header =
                            "--" + boundary + "\r\n"
                            "Content-Type: image/jpeg\r\n"
                            "Content-Length: " + std::to_string(jpeg->size()) + "\r\n\r\n";
                        if (!sink.write(header.data(), header.size())) return false;
                        if (!sink.write(reinterpret_cast<const char*>(jpeg->data()), jpeg->size())) return false;
                        if (!sink.write("\r\n", 2)) return false;
                    }
                    sink.done();
                    return true;
                });
        });

        impl_->svr.Post(R"(/webrtc/whep/(.+))", [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            if (req.matches.size() < 2) { res.status = 400; return; }
            if (req.get_header_value("Content-Type").find("application/sdp") == std::string::npos) {
                res.status = 415;
                res.set_content("Content-Type must be application/sdp", "text/plain");
                return;
            }
            const std::string key = req.matches[1];
            if (!webrtc_) {
                res.status = 503;
                res.set_content("WebRTC publisher unavailable", "text/plain");
                return;
            }
            auto result = webrtc_->handle_offer(key, req.body);
            res.status = result.status;
            if (!result.location.empty()) res.set_header("Location", result.location);
            res.set_content(result.body, result.content_type);
        });

        impl_->svr.Delete(R"(/webrtc/whep/sessions/(.+))",
                          [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            if (!webrtc_ || req.matches.size() < 2) { res.status = 404; return; }
            res.status = webrtc_->delete_session(req.matches[1]) ? 204 : 404;
        });

        impl_->svr.Patch(R"(/webrtc/whep/sessions/(.+))",
                         [this, origin](const httplib::Request& req, httplib::Response& res) {
            apply_cors_(origin(req), res);
            res.status = 501;
            res.set_content("trickle ICE is not implemented", "text/plain");
        });

        server_thread_ = std::thread([this] {
            std::cout << "[Runner HTTP] Health: http://" << host_ << ":" << port_ << "/health\n";
            std::cout << "[Runner HTTP] Video: http://" << host_ << ":" << port_ << "/video/<stream_id>/<profile>\n";
            std::cout << "[Runner HTTP] WHEP: http://" << host_ << ":" << port_ << "/webrtc/whep/<stream_id>/<profile>\n";
            impl_->svr.listen(host_.c_str(), port_);
        });
        return true;
    }

    void RunnerHTTPServer::stop() {
        if (!running_) return;
        running_ = false;
        {
            std::lock_guard lk(streams_mtx_);
            for (auto& kv : streams_) kv.second->cv.notify_all();
        }
        if (impl_) impl_->svr.stop();
        if (server_thread_.joinable()) server_thread_.join();
    }
}
