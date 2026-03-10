#pragma once

#include <condition_variable>
#include <deque>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace ss {
    template <class T>
    class BoundedQueue {
    public:
        explicit BoundedQueue(size_t capacity) : cap_(capacity) {}

        void push_drop_oldest(T v) {
            {
                std::lock_guard lk(m_);
                if (stopped_ || cap_ == 0) return;
                if (q_.size() >= cap_) {
                    q_.pop_front();
                    ++dropped_count_;
                }
                q_.push_back(std::move(v));
            }
            cv_.notify_one();
        }

        bool try_pop(T& out) {
            std::lock_guard lk(m_);
            if (q_.empty()) return false;
            out = std::move(q_.front());
            q_.pop_front();
            return true;
        }

        bool pop_for(T& out, std::chrono::milliseconds d) {
            std::unique_lock lk(m_);
            if (!cv_.wait_for(lk, d, [&]{ return stopped_ || !q_.empty(); })) return false;
            if (stopped_ || q_.empty()) return false;
            out = std::move(q_.front());
            q_.pop_front();
            return true;
        }

        void stop() {
            {
                std::lock_guard lk(m_);
                stopped_ = true;
            }
            cv_.notify_all();
        }

        void reset() {
            {
                std::lock_guard lk(m_);
                q_.clear();
                dropped_count_ = 0;
                stopped_ = false;
            }
            cv_.notify_all();
        }

        size_t size() const {
            std::lock_guard lk(m_);
            return q_.size();
        }

        size_t capacity() const { return cap_; }

        uint64_t dropped_count() const {
            std::lock_guard lk(m_);
            return dropped_count_;
        }
    private:
        size_t cap_;
        mutable std::mutex m_;
        std::condition_variable cv_;
        std::deque<T> q_;
        bool stopped_ = false;
        uint64_t dropped_count_ = 0;
    };
}
