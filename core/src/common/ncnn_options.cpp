#include <common/ncnn_options.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace veilsight {
    namespace {
        bool parse_truthy(const char* value) {
            if (value == nullptr) return false;

            std::string normalized(value);
            std::transform(normalized.begin(),
                           normalized.end(),
                           normalized.begin(),
                           [](unsigned char ch) {
                               return static_cast<char>(std::tolower(ch));
                           });

            return normalized == "1" ||
                   normalized == "true" ||
                   normalized == "yes" ||
                   normalized == "on";
        }
    }

    bool ncnn_vulkan_requested_from_env() {
        return parse_truthy(std::getenv("VEILSIGHT_NCNN_VULKAN"));
    }

    void configure_ncnn_net(ncnn::Net& net, int num_threads) {
#if NCNN_VULKAN
        net.opt.use_vulkan_compute = ncnn_vulkan_requested_from_env();
#else
        net.opt.use_vulkan_compute = false;
#endif
        net.opt.num_threads = std::max(1, num_threads);
    }
}
