#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <condition_variable>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <functional>

class threadpool {
private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool active = true;

public:
    threadpool(size_t num_threads) {
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this] {
                            return !tasks.empty() || !active;
                        });
                        if (!active && tasks.empty()) {
                            return;
                        }
                        task = move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    ~threadpool() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            active = false;
        }
        cv.notify_all();
        for (auto &t : threads) {
            t.join();
        }
    }
    void enqueue(std::function<void()> task) {
        std::unique_lock<std::mutex> lock(mtx);
        tasks.emplace(move(task));
    }
};

#endif