#pragma once

#include <pipeline/types.hpp>

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace veilsight::render {
    struct RenderRule {
        std::string id;
        std::string kind;
        std::string name;
        bool enabled = true;
        std::vector<PointF> points;
        int64_t min_gap_ms = 1000;
        double dwell_threshold_s = 0.0;
    };

    struct RenderEvent {
        int64_t frame_id = 0;
        int64_t pts_ns = 0;
        std::string rule_id;
        std::string rule_name;
        std::string kind;
        int track_id = -1;
        PointF position;
        std::string direction;
        std::map<std::string, double> payload_numbers;
    };

    class RenderRuleEngine {
    public:
        explicit RenderRuleEngine(std::vector<RenderRule> rules = {});

        void set_rules(std::vector<RenderRule> rules);
        const std::vector<RenderRule>& rules() const;

        std::vector<RenderEvent> process_frame(int64_t frame_id,
                                               int64_t pts_ns,
                                               const std::vector<Box>& tracks);

    private:
        struct TrackRuleState {
            std::optional<PointF> previous_position;
            std::optional<double> previous_side;
            std::optional<int64_t> last_line_event_ms;
            bool inside = false;
            int64_t entered_pts_ns = 0;
            bool dwell_emitted = false;
        };

        std::vector<RenderRule> rules_;
        std::map<std::string, std::map<int, TrackRuleState>> state_;
    };

    std::vector<RenderRule> load_render_rules_yaml(const std::string& path);
    std::string render_event_to_json(const RenderEvent& event);
}
