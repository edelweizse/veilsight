#pragma once

#include <common/config.hpp>
#include <pipeline/types.hpp>

#include <memory>
#include <vector>

namespace ss {
    class IRecognizer {
    public:
        virtual ~IRecognizer() = default;
        virtual void annotate(FrameCtx& frame, std::vector<Box>& tracks) = 0;
    };

    class IRecognizerFactory {
    public:
        virtual ~IRecognizerFactory() = default;
        virtual std::unique_ptr<IRecognizer> create() const = 0;
        virtual int backend_threads() const = 0;
    };

    std::unique_ptr<IRecognizerFactory> create_recognizer_factory(const RecognizerModuleConfig& cfg);
    std::unique_ptr<IRecognizer> create_recognizer(const RecognizerModuleConfig& cfg);
}
