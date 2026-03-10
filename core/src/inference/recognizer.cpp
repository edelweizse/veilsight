#include <inference/recognizer.hpp>

#include <algorithm>
#include <stdexcept>

namespace ss {
    namespace {
        class NoopRecognizer final : public IRecognizer {
        public:
            void annotate(FrameCtx&, std::vector<Box>&) override {}
        };

        class NoopRecognizerFactory final : public IRecognizerFactory {
        public:
            std::unique_ptr<IRecognizer> create() const override {
                return std::make_unique<NoopRecognizer>();
            }

            int backend_threads() const override {
                return 1;
            }
        };
    }

    std::unique_ptr<IRecognizerFactory> create_recognizer_factory(const RecognizerModuleConfig& cfg) {
        if (cfg.type.empty() || cfg.type == "none") {
            return std::make_unique<NoopRecognizerFactory>();
        }
        throw std::invalid_argument("[Recognizer] Unsupported recognizer type: " + cfg.type);
    }

    std::unique_ptr<IRecognizer> create_recognizer(const RecognizerModuleConfig& cfg) {
        return create_recognizer_factory(cfg)->create();
    }
}
