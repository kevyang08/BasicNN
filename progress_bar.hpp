#pragma once

#include <atomic>
#include <functional>
#include <future>
#include <iostream>
#include <thread>

struct ProgressBar {
    virtual void update_epoch(const int cur_epoch) const = 0;
    virtual void update_progress(const int cur_progress) = 0;
    virtual void start_print_progress() = 0;
    virtual void end_print_progress() = 0;
};

struct VerboseProgressBar : ProgressBar {
    const int total_epochs;
    const int data_size;
    std::atomic<int> progress;
    std::future<void> future_bar;
    VerboseProgressBar(const int total_epochs, const int data_size) : total_epochs{total_epochs}, data_size{data_size}, progress{0} {}
    inline void update_epoch(const int cur_epoch) const {
        std::cout << "Epoch " << cur_epoch << "/" << total_epochs << std::endl;
    }
    inline void update_progress(const int cur_progress) {
        progress.store(cur_progress);
    }
    void start_print_progress() {
        future_bar = std::async(std::launch::async, [this]() {
            while (progress.load() + 1 < data_size) {
                const int cur_progress = (int)((progress.load() + 1.0)/data_size * 100);
                std::cout << "\r[";
                for (int i = 0; i < 50; i++) std::cout << (cur_progress > i * 2 + 1 ? '=' : ' ');
                std::cout << "] " << cur_progress << "%";
            }
            std::cout << "\r[";
            for (int i = 0; i < 50; i++) std::cout << '=';
            std::cout << "] 100%" << std::endl;
        });
    }
    inline void end_print_progress() {
        future_bar.get();
    }
};

struct NullProgressBar : ProgressBar {
    inline void update_epoch(const int cur_epoch) const {}
    inline void update_progress(const int cur_progress) {}
    inline void start_print_progress() {}
    inline void end_print_progress() {}
};