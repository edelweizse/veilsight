#include <encode/mjpeg_server.hpp>
#include <iostream>
#include <sstream>
#include <httplib.h>
#include <opencv2/imgcodecs.hpp>

namespace ss {
    struct MJPEGServer:: Impl {
        httplib::Server svr;
    };

    MJPEGServer::MJPEGServer(std::string host, int port)
        : impl_(new Impl()),
          host_(std::move(host)),
          port_(port) {}

    MJPEGServer::~MJPEGServer() {
        stop();
        delete impl_;
    }

    std::shared_ptr<MJPEGServer::StreamState> MJPEGServer::get_or_create_(const std::string& key) const {
        std::lock_guard lk(streams_mtx_);
        auto& p = streams_[key];
        if (!p) p = std::make_shared<StreamState>();
        return p;
    }

    std::shared_ptr<MJPEGServer::StreamState> MJPEGServer::get_(const std::string& key) const {
        std::lock_guard lk(streams_mtx_);
        auto it = streams_.find(key);
        if (it == streams_.end()) return nullptr;
        return it->second;
    }

    void MJPEGServer::push_jpeg(const std::string& stream_key,
                                std::shared_ptr<const std::vector<uint8_t>> jpeg) {
        auto st = get_or_create_(stream_key);
        {
            std::lock_guard lk(st->mtx);
            st->last_jpeg = std::move(jpeg);
            ++st->seq;
        }
        st->cv.notify_all();
    }

    void MJPEGServer::push_jpeg(const std::string& stream_key, const cv::Mat& frame, int quality) {
        if (frame.empty() || frame.type() != CV_8UC3) { return; }
        std::vector<uint8_t> tmp;
        std::vector params = {cv::IMWRITE_JPEG_QUALITY, quality};

        if (!cv::imencode(".jpg", frame, tmp, params)) return;

        auto bytes = std::make_shared<std::vector<uint8_t>>(std::move(tmp));
        push_jpeg(stream_key, bytes);
    }

    void MJPEGServer::push_meta(const std::string& stream_key, std::string json) {
        auto st = get_or_create_(stream_key);
        std::lock_guard<std::mutex> lk(st->meta_mtx);
        st->last_meta = std::move(json);
    }

    void MJPEGServer::push_metrics(std::string json) {
        std::lock_guard<std::mutex> lk(metrics_mtx_);
        last_metrics_json_ = std::move(json);
    }

    std::vector<std::string> MJPEGServer::list_streams() const {
        std::lock_guard lk(streams_mtx_);
        std::vector<std::string> out;
        out.reserve(streams_.size());
        for (const auto& kv : streams_) out.push_back(kv.first);
        std::sort(out.begin(), out.end());
        return out;
    }

    void MJPEGServer::register_stream(const std::string& stream_key) {
        (void)get_or_create_(stream_key);
    }

    bool MJPEGServer::start() {
        if (running_) return true;
        running_ = true;

        // /streams -> JSON list of stream keys
        impl_->svr.Get("/streams", [this](const httplib::Request&, httplib::Response& res) {
           auto keys = list_streams();
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < keys.size(); ++i) {
                oss << "\"" << keys[i] << "\"";
                if (i + 1 < keys.size()) oss << ",";
            }
            oss << "]";
            res.set_content(oss.str(), "application/json");
            res.set_header("Cache-Control", "no-cache");
        });

        // /meta/<stream_key>
        impl_->svr.Get(R"(/meta/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
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

        // /snapshot/<stream_key> -> last_jpeg once
        impl_->svr.Get(R"(/snapshot/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
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
            res.set_content(reinterpret_cast<const char *>(jpeg->data()), jpeg->size(), "image/jpeg");
            res.set_header("Cache-Control", "no-cache");
        });

        // /video/<stream_key> -> MJPEG
        impl_->svr.Get(R"(/video/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            if (req.matches.size() < 2) { res.status = 400; return; }
            const std::string key = req.matches[1];

            auto st = get_(key);
            if (!st) { res.status = 404; return; }

            res.set_header("Cache-Control", "no-cache");
            res.set_header("Pragma", "no-cache");
            res.set_header("Connection", "close");

            const std::string boundary = "frame";
            res.set_header("Content-Type", "multipart/x-mixed-replace; boundary=" + boundary);

            res.set_chunked_content_provider(
                "multipart/x-mixed-replace; boundary=" + boundary,
                [this, st, boundary](size_t /*offset*/, httplib::DataSink& sink) {
                    uint64_t last_sent = 0;
                    {
                        std::unique_lock lk(st->mtx);
                        st->cv.wait(lk, [&] { return st->seq != 0 || !running_; });
                        if (!running_) { sink.done(); return true; }
                        last_sent = st->seq;;
                    }

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
                        if (!sink.write(reinterpret_cast<const char *>(jpeg->data()), jpeg->size())) return false;
                        if (!sink.write("\r\n", 2)) return false;
                    }

                    sink.done();
                    return true;
                }
            );
        });

        impl_->svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("ok", "text/plain");
        });

        impl_->svr.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
            std::string payload;
            {
                std::lock_guard<std::mutex> lk(metrics_mtx_);
                payload = last_metrics_json_.empty() ? "{}" : last_metrics_json_;
            }
            res.set_content(payload, "application/json");
            res.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
            res.set_header("Pragma", "no-cache");
        });

        server_thread_ = std::thread([this] {
            std::cout << "[MJPEG] Streams list: http://" << host_ << ":" << port_ << "/streams\n";
            std::cout << "[MJPEG] Video: http://" << host_ << ":" << port_ << "/video/<stream_id>/<profile>\n";
            impl_->svr.listen(host_.c_str(), port_);
        });

        return true;
    }

    void MJPEGServer::stop() {
        if (!running_) return;
        running_ = false;

        {
            std::lock_guard lk(streams_mtx_);
            for (auto& kv : streams_) {
                kv.second->cv.notify_all();
            }
        }

        if (impl_) impl_->svr.stop();
        if (server_thread_.joinable()) server_thread_.join();
    }
}
