#pragma once

#include <ncnn/net.h>

namespace veilsight {
    bool ncnn_vulkan_requested_from_env();
    void configure_ncnn_net(ncnn::Net& net, int num_threads);
}
