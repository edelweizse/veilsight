#pragma once

#include <common/config.hpp>
#include <pipeline/types.hpp>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace veilsight {
    struct SceneGridAssociationCost {
        float raw_cost = 0.0f;
        float extra_cost = 0.0f;
    };

    class SceneGrid {
    public:
        explicit SceneGrid(SceneGridConfig cfg = {});

        const SceneGridConfig& config() const { return cfg_; }

        void begin_frame(const std::string& stream_id);
        void observe(const std::string& stream_id, int width, int height, const Box& box);
        void observe_transition(const std::string& stream_id,
                                int width,
                                int height,
                                const Box& from,
                                const Box& to);

        std::pair<int, int> cell_for_box(int width, int height, const Box& box) const;
        SceneGridAssociationCost association_cost(const std::string& stream_id,
                                                  int width,
                                                  int height,
                                                  const Box& track,
                                                  const Box& detection) const;

    private:
        struct StreamState {
            int frames_seen = 0;
            std::vector<float> occupancy;
            std::vector<float> transitions;
        };

        int cell_index_(int row, int col) const;
        StreamState& state_(const std::string& stream_id);
        const StreamState* find_state_(const std::string& stream_id) const;

        SceneGridConfig cfg_;
        std::unordered_map<std::string, StreamState> streams_;
    };
}
