#pragma once

namespace veilsight {
    class IPipelineLifecycle {
    public:
        virtual ~IPipelineLifecycle() = default;
        virtual bool start() = 0;
        virtual void stop() = 0;
        virtual bool is_running() const = 0;
    };
}
