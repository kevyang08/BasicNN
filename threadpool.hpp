#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <condition_variable>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <functional>
#include <iostream>

class threadpool {
private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv_tasks, cv_done;
    bool active = true;
    size_t cur_tasks = 0;

public:
    threadpool(size_t num_threads) {
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv_tasks.wait(lock, [this] {
                            return !tasks.empty() || !active;
                        });
                        if (!active && tasks.empty()) {
                            return;
                        }
                        task = move(tasks.front());
                        tasks.pop();
                    }
                    try {
                        task();
                    } catch (std::exception e) {
                        std::cerr << "Caught exception: " << e.what() << std::endl;
                    }
                    std::unique_lock<std::mutex> lock(mtx);
                    --cur_tasks;
                    if (!cur_tasks)
                        cv_done.notify_one();
                }
            });
        }
    }
    ~threadpool() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            active = false;
        }
        cv_tasks.notify_all();
        for (auto &t : threads) {
            t.join();
        }
    }
    void enqueue(std::function<void()> task) {
        std::unique_lock<std::mutex> lock(mtx);
        ++cur_tasks;
        tasks.emplace(move(task));
        cv_tasks.notify_one();
    }
    void wait_all() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_done.wait(lock, [this] {
            return !cur_tasks;
        });
    }
};

#endif