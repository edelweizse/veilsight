#include <render/rules.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {
    int g_failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++g_failures;
            std::cerr << "[FAIL] " << message << "\n";
        }
    }

    veilsight::Box track(int id, float foot_x, float foot_y = 10.0f) {
        veilsight::Box box;
        box.id = id;
        box.x = foot_x - 5.0f;
        box.y = foot_y - 10.0f;
        box.w = 10.0f;
        box.h = 10.0f;
        box.score = 0.9f;
        return box;
    }

    veilsight::render::RenderRule line_rule() {
        veilsight::render::RenderRule rule;
        rule.id = "line";
        rule.kind = "line";
        rule.name = "Line";
        rule.points = {{50.0f, 0.0f}, {50.0f, 100.0f}};
        rule.min_gap_ms = 1000;
        return rule;
    }

    veilsight::render::RenderRule area_rule() {
        veilsight::render::RenderRule rule;
        rule.id = "area";
        rule.kind = "area";
        rule.name = "Area";
        rule.points = {{0.0f, 0.0f}, {100.0f, 0.0f}, {100.0f, 100.0f}, {0.0f, 100.0f}};
        rule.dwell_threshold_s = 1.0;
        return rule;
    }

    void test_line_crossing_emits_once() {
        veilsight::render::RenderRuleEngine engine({line_rule()});
        std::vector<veilsight::render::RenderEvent> events;
        auto first = engine.process_frame(1, 0, {track(1, 35.0f)});
        auto second = engine.process_frame(2, 1'000'000'000, {track(1, 55.0f)});
        events.insert(events.end(), first.begin(), first.end());
        events.insert(events.end(), second.begin(), second.end());

        check(events.size() == 1, "line crossing should emit one event");
        check(events[0].kind == "line_cross", "line event kind should be line_cross");
        check(events[0].track_id == 1, "line event should keep track id");
        check(!events[0].direction.empty(), "line event should include direction");
    }

    void test_line_debounce_suppresses_duplicate() {
        veilsight::render::RenderRuleEngine engine({line_rule()});
        std::vector<veilsight::render::RenderEvent> events;
        for (auto out : {
                 engine.process_frame(1, 0, {track(1, 35.0f)}),
                 engine.process_frame(2, 500'000'000, {track(1, 55.0f)}),
                 engine.process_frame(3, 700'000'000, {track(1, 35.0f)}),
             }) {
            events.insert(events.end(), out.begin(), out.end());
        }

        check(events.size() == 1, "line debounce should suppress duplicate crossing inside min_gap_ms");
    }

    void test_area_enter_exit_and_dwell() {
        veilsight::render::RenderRuleEngine engine({area_rule()});
        std::vector<veilsight::render::RenderEvent> events;
        for (auto out : {
                 engine.process_frame(1, 0, {track(1, 125.0f)}),
                 engine.process_frame(2, 1'000'000'000, {track(1, 20.0f)}),
                 engine.process_frame(3, 2'100'000'000, {track(1, 22.0f)}),
                 engine.process_frame(4, 3'000'000'000, {track(1, 125.0f)}),
             }) {
            events.insert(events.end(), out.begin(), out.end());
        }

        check(events.size() == 3, "area rule should emit enter, dwell threshold, and exit");
        check(events[0].kind == "area_enter", "first area event should be enter");
        check(events[1].kind == "area_dwell_threshold", "second area event should be dwell threshold");
        check(events[2].kind == "area_exit", "third area event should be exit");
    }

    void test_disabled_rules_are_ignored() {
        auto rule = line_rule();
        rule.enabled = false;
        veilsight::render::RenderRuleEngine engine({rule});
        engine.process_frame(1, 0, {track(1, 35.0f)});
        const auto events = engine.process_frame(2, 1'000'000'000, {track(1, 55.0f)});
        check(events.empty(), "disabled rules should be ignored");
    }
}

int main() {
    test_line_crossing_emits_once();
    test_line_debounce_suppresses_duplicate();
    test_area_enter_exit_and_dwell();
    test_disabled_rules_are_ignored();

    if (g_failures) {
        std::cerr << g_failures << " render rule test(s) failed\n";
        return 1;
    }
    std::cout << "render rule tests passed\n";
    return 0;
}
