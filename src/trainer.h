# pragma once
#include <atomic>
#include <future>
#include <mutex>
#include <iostream>
#include <thread>
#include <chrono>
#include <Eigen/Dense>

#include "neural_net.h"

class Trainer {
public:
    Trainer(Neural_net& neural_net) : neural_net(neural_net){}

    void start_training(Eigen::MatrixXd &X, Eigen::RowVectorXd &Y, double alpha) {
        if (training_flag) {
            return;
        }
        training_flag = true;
        task = std::async(std::launch::async, [this, &X, &Y, alpha]()     {
            while (training_flag) {
                is_training_flag = true;
                neural_net.gradient_descent(X, Y, alpha);
                iterations = iterations + 1;

                if (iterations % accuracy_interval == 0) {
                    std::lock_guard<std::mutex> lock(test_data_mutex);
                    if (X_test.size()!=0) {
                        accuracy = neural_net._get_accuracy(X_test, Y_test);
                    }
                    else {
                        accuracy = neural_net._get_accuracy(X, Y);
                    }
                }
            }
            is_training_flag = false;

        });
    }

    void stop_training() {
        training_flag = false;
    }

    // Accuracy Callbacks
    void set_accuracy_callback() {  // Default callback, uses training data to check accuracy.
        std::lock_guard<std::mutex> lock(test_data_mutex);
        X_test = Eigen::MatrixXd();
        Y_test = Eigen::RowVectorXd();
        accuracy_interval = 5;
    }
    void set_accuracy_callback(Eigen::MatrixXd& X, Eigen::RowVectorXd& Y, int accuracy_interval) { // Set testing data
        std::lock_guard<std::mutex> lock(test_data_mutex);
        X_test = X;
        Y_test = Y;
        this->accuracy_interval = accuracy_interval;
    }

    double get_accuracy(Eigen::MatrixXd& X, Eigen::RowVectorXd& Y) const {
        return accuracy;
    }

    void wait() {
        if (task.valid())
            task.get();
    }

    bool is_training() {return is_training_flag;}
    int get_iterations() {return iterations;}

private:
    Neural_net& neural_net;
    std::atomic<bool> training_flag{false}; // Whether to start/stop training
    std::atomic<bool> is_training_flag{false};  // Tells external methods whether training is actually start/stopped
    std::atomic<int> iterations{0};
    std::future<void> task;

    Eigen::MatrixXd X_test;
    Eigen::RowVectorXd Y_test;
    int accuracy_interval = 5;
    std::atomic<double> accuracy{0.0};
    std::mutex test_data_mutex;
};