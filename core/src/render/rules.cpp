#include <render/rules.hpp>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace veilsight::render {
    namespace {
        constexpr double kEpsilon = 1e-5;

        PointF track_position(const Box& track) {
            return PointF{track.x + track.w * 0.5f, track.y + track.h};
        }

        double signed_line_distance(const PointF& a, const PointF& b, const PointF& p) {
            return static_cast<double>(b.x - a.x) * static_cast<double>(p.y - a.y) -
                   static_cast<double>(b.y - a.y) * static_cast<double>(p.x - a.x);
        }

        bool point_in_polygon(const std::vector<PointF>& polygon, const PointF& point) {
            if (polygon.size() < 3) return false;
            bool inside = false;
            for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
                const PointF& pi = polygon[i];
                const PointF& pj = polygon[j];
                const bool crosses = ((pi.y > point.y) != (pj.y > point.y)) &&
                                     (point.x < (pj.x - pi.x) * (point.y - pi.y) /
                                                    ((pj.y - pi.y) == 0.0f ? 1e-6f : (pj.y - pi.y)) +
                                                pi.x);
                if (crosses) inside = !inside;
            }
            return inside;
        }

        int64_t pts_to_ms(int64_t pts_ns) {
            return pts_ns / 1000000;
        }

        RenderEvent base_event(const RenderRule& rule,
                               const std::string& kind,
                               int64_t frame_id,
                               int64_t pts_ns,
                               const Box& track,
                               const PointF& position) {
            RenderEvent event;
            event.frame_id = frame_id;
            event.pts_ns = pts_ns;
            event.rule_id = rule.id;
            event.rule_name = rule.name;
            event.kind = kind;
            event.track_id = track.id;
            event.position = position;
            return event;
        }

        PointF yaml_point(const YAML::Node& node) {
            PointF point;
            point.x = node["x"] ? node["x"].as<float>() : 0.0f;
            point.y = node["y"] ? node["y"].as<float>() : 0.0f;
            return point;
        }

        std::string json_escape(const std::string& value) {
            std::ostringstream out;
            for (char ch : value) {
                switch (ch) {
                    case '\\': out << "\\\\"; break;
                    case '"': out << "\\\""; break;
                    case '\n': out << "\\n"; break;
                    case '\r': out << "\\r"; break;
                    case '\t': out << "\\t"; break;
                    default: out << ch; break;
                }
            }
            return out.str();
        }
    }

    RenderRuleEngine::RenderRuleEngine(std::vector<RenderRule> rules)
        : rules_(std::move(rules)) {}

    void RenderRuleEngine::set_rules(std::vector<RenderRule> rules) {
        rules_ = std::move(rules);
        state_.clear();
    }

    const std::vector<RenderRule>& RenderRuleEngine::rules() const {
        return rules_;
    }

    std::vector<RenderEvent> RenderRuleEngine::process_frame(int64_t frame_id,
                                                             int64_t pts_ns,
                                                             const std::vector<Box>& tracks) {
        std::vector<RenderEvent> events;
        for (const RenderRule& rule : rules_) {
            if (!rule.enabled) continue;
            if (rule.kind == "line" && rule.points.size() >= 2) {
                const PointF a = rule.points[0];
                const PointF b = rule.points[1];
                for (const Box& track : tracks) {
                    if (track.id < 0) continue;
                    const PointF position = track_position(track);
                    const double side = signed_line_distance(a, b, position);
                    auto& state = state_[rule.id][track.id];
                    if (state.previous_side.has_value() &&
                        std::fabs(side) > kEpsilon &&
                        std::fabs(*state.previous_side) > kEpsilon &&
                        ((*state.previous_side > 0.0) != (side > 0.0))) {
                        const int64_t now_ms = pts_to_ms(pts_ns);
                        if (!state.last_line_event_ms.has_value() ||
                            now_ms - *state.last_line_event_ms >= rule.min_gap_ms) {
                            RenderEvent event = base_event(rule, "line_cross", frame_id, pts_ns, track, position);
                            event.direction = *state.previous_side > 0.0
                                                  ? "positive_to_negative"
                                                  : "negative_to_positive";
                            events.push_back(std::move(event));
                            state.last_line_event_ms = now_ms;
                        }
                    }
                    state.previous_position = position;
                    if (std::fabs(side) > kEpsilon) state.previous_side = side;
                }
            } else if (rule.kind == "area" && rule.points.size() >= 3) {
                for (const Box& track : tracks) {
                    if (track.id < 0) continue;
                    const PointF position = track_position(track);
                    const bool inside = point_in_polygon(rule.points, position);
                    auto& state = state_[rule.id][track.id];

                    if (inside && !state.inside) {
                        state.inside = true;
                        state.entered_pts_ns = pts_ns;
                        state.dwell_emitted = false;
                        events.push_back(base_event(rule, "area_enter", frame_id, pts_ns, track, position));
                    } else if (!inside && state.inside) {
                        RenderEvent event = base_event(rule, "area_exit", frame_id, pts_ns, track, position);
                        event.payload_numbers["total_dwell_s"] =
                            static_cast<double>(pts_ns - state.entered_pts_ns) / 1000000000.0;
                        events.push_back(std::move(event));
                        state.inside = false;
                        state.dwell_emitted = false;
                    } else if (inside && rule.dwell_threshold_s > 0.0 && !state.dwell_emitted) {
                        const double dwell_s = static_cast<double>(pts_ns - state.entered_pts_ns) / 1000000000.0;
                        if (dwell_s >= rule.dwell_threshold_s) {
                            RenderEvent event =
                                base_event(rule, "area_dwell_threshold", frame_id, pts_ns, track, position);
                            event.payload_numbers["dwell_s"] = dwell_s;
                            events.push_back(std::move(event));
                            state.dwell_emitted = true;
                        }
                    }
                    state.previous_position = position;
                }
            }
        }
        return events;
    }

    std::vector<RenderRule> load_render_rules_yaml(const std::string& path) {
        if (path.empty()) return {};
        const YAML::Node root = YAML::LoadFile(path);
        const YAML::Node rules_node = root["rules"];
        if (!rules_node) return {};
        if (!rules_node.IsSequence()) throw std::runtime_error("render rules YAML field 'rules' must be a sequence");

        std::vector<RenderRule> rules;
        for (size_t i = 0; i < rules_node.size(); ++i) {
            const YAML::Node item = rules_node[i];
            RenderRule rule;
            rule.id = item["id"] ? item["id"].as<std::string>() : "rule_" + std::to_string(i + 1);
            rule.kind = item["kind"] ? item["kind"].as<std::string>() : "";
            rule.name = item["name"] ? item["name"].as<std::string>() : rule.id;
            rule.enabled = item["enabled"] ? item["enabled"].as<bool>() : true;

            const YAML::Node points = item["geometry"]["points"];
            if (points && points.IsSequence()) {
                for (const auto& point : points) rule.points.push_back(yaml_point(point));
            }

            const YAML::Node settings = item["settings"];
            if (settings) {
                if (settings["min_gap_ms"]) rule.min_gap_ms = settings["min_gap_ms"].as<int64_t>();
                if (settings["dwell_threshold_s"]) rule.dwell_threshold_s = settings["dwell_threshold_s"].as<double>();
            }
            rules.push_back(std::move(rule));
        }
        return rules;
    }

    std::string render_event_to_json(const RenderEvent& event) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(3);
        out << "{";
        out << "\"frame_id\":" << event.frame_id << ",";
        out << "\"pts_ns\":" << event.pts_ns << ",";
        out << "\"rule_id\":\"" << json_escape(event.rule_id) << "\",";
        out << "\"rule_name\":\"" << json_escape(event.rule_name) << "\",";
        out << "\"kind\":\"" << json_escape(event.kind) << "\",";
        out << "\"track_id\":" << event.track_id << ",";
        out << "\"position\":{\"x\":" << event.position.x << ",\"y\":" << event.position.y << "}";
        if (!event.direction.empty()) out << ",\"direction\":\"" << json_escape(event.direction) << "\"";
        out << ",\"payload\":{";
        bool first = true;
        for (const auto& [key, value] : event.payload_numbers) {
            if (!first) out << ",";
            first = false;
            out << "\"" << json_escape(key) << "\":" << value;
        }
        out << "}}";
        return out.str();
    }
}
