#pragma once
#include <Eigen/Dense>

struct Layer {				// Struct represents a layer
    Eigen::MatrixXd W;	// Weights
    Eigen::MatrixXd Z;  // Pre-Activation data
    Eigen::MatrixXd A; 	// Activation data
    Eigen::VectorXd b;	// Biases

    Layer(int input_neurons, int output_neurons, int m) {
        W = Eigen::MatrixXd::Random(output_neurons, input_neurons) * 0.5f;
        Z = Eigen::MatrixXd::Zero(output_neurons, m);
        A = Eigen::MatrixXd::Zero(output_neurons, m);
        b = Eigen::VectorXd::Random(output_neurons) * 0.5f;

    }
};

struct Gradient {
    Eigen::MatrixXd dZ;
    Eigen::MatrixXd dW;
    Eigen::MatrixXd db;

};

namespace activations {
    inline Eigen::MatrixXd ReLU(const Eigen::MatrixXd& Z) {
        return Z.array().max(0.0);
    }

    inline Eigen::MatrixXd deriv_ReLU(const Eigen::MatrixXd& Z) {
        return (Z.array() > 0).cast<double>();

    }

    inline Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z) {
        // max per column (row vector)
        Eigen::RowVectorXd max_per_col = Z.colwise().maxCoeff();

        // subtract max per column from each column (broadcasting)
        Eigen::MatrixXd Z_stable = Z;
        for (int i = 0; i < Z.cols(); ++i) {
            Z_stable.col(i).array() -= max_per_col(i);
        }

        Eigen::MatrixXd exp_Z = Z_stable.array().exp();

        // sum of exp per column
        Eigen::RowVectorXd sum_exp = exp_Z.colwise().sum();

        // divide each column by sum of its exponential
        Eigen::MatrixXd A = exp_Z;
        for (int i = 0; i < exp_Z.cols(); ++i) {
            A.col(i).array() /= sum_exp(i);
        }

        return A;
    }
}

class Neural_net {
    public:

    Neural_net(const std::vector<std::pair<int, int>>& layer_data, int m) {
        for (const auto & i : layer_data) {
            layers.emplace_back(std::get<0>(i), std::get<1>(i), m);
        }
    }

    void gradient_descent(Eigen::MatrixXd& X,  Eigen::RowVectorXd& Y, double alpha) {
        forward_prop(X);
        back_prop(X, Y, alpha);
    }


    double get_accuracy(const Eigen::MatrixXd& Y) {
        return get_accuracy(get_predictions(), Y.cast<int>());
    }

    double _get_accuracy(Eigen::MatrixXd &X, Eigen::RowVectorXd &Y) {

        const int m = Y.size();
        Eigen::RowVectorXi predictions = get_predictions();
        int correct_predictions = 0;

        forward_prop(X);
        predictions = get_predictions();

        for (int i=0; i < m; i++) {
            if (predictions[i] == Y[i]) {
                correct_predictions++;
            }
        }

        return static_cast<double>(correct_predictions) / static_cast<double>(m);

    }

    int make_prediction(Eigen::VectorXd& X) {
         Eigen::MatrixXd input = X;
         input.resize(X.size(), 1);

        // Forward Propagation
        for (size_t j=0; j < layers.size() - 1; j++) {
            forward_prop_hidden_layer(layers[j], input);
            input = layers[j].A;
        }
        forward_prop_output(layers[layers.size() - 1], input);

        Eigen::RowVectorXi result = get_predictions();

        return result(0);

    }

private:
    std::vector<Layer> layers;

    Eigen::MatrixXd one_hot(const Eigen::RowVectorXd& Y) { // Returns one hot version of labels

        int num_classes = static_cast<int>(Y.maxCoeff()) + 1 ;
        int n = Y.size();

        Eigen::MatrixXd Y_one_hot = Eigen::MatrixXd::Zero(n, num_classes);

        for (int i = 0; i < n; ++i) {
            Y_one_hot(i, static_cast<int>(Y(i))) = 1;
        }

        Y_one_hot.transposeInPlace();

        return Y_one_hot;
    }

    //  Forward prop function for hidden layer
    void forward_prop_hidden_layer(Layer& layer, const Eigen::MatrixXd& X) {
        Eigen::MatrixXd Z = (layer.W * X);
        Z.colwise() += layer.b;

        layer.Z = Z;
        layer.A = activations::ReLU(Z);
    }

    // Forward prop function for output layer
    void forward_prop_output(Layer& layer, const Eigen::MatrixXd& X){
        Eigen::MatrixXd Z = (layer.W * X);
        Z.colwise() += layer.b;

        layer.Z = Z;
        layer.A = activations::softmax(Z);
    }

    // Back prop function for hidden layer
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
    back_prop_hidden_layer(const Layer& layer, const Layer& post_layer,
                 const Gradient& gradient,
                 const Eigen::MatrixXd& X) { // X is the activation Matrix from previous layer

        const int m = X.cols();
        Eigen::MatrixXd dZ = (post_layer.W.transpose() * gradient.dZ).array() * activations::deriv_ReLU(layer.Z).array();
        Eigen::MatrixXd dW = (1.0 / m) * (dZ * X.transpose());
        Eigen::MatrixXd db = (1.0 / m) * dZ.rowwise().sum();

        return std::make_tuple(dW, db, dZ);

    }

    // Back prop function for output layer
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
    back_prop_output(const Layer& prev_layer, const Layer& layer,
                     const Eigen::RowVectorXd& Y){
        int m = Y.size();
        Eigen::MatrixXd dZ = layer.A - one_hot(Y);
        Eigen::MatrixXd dW = (1.0 / m) * (dZ * prev_layer.A.transpose());
        Eigen::MatrixXd db = (1.0 / m) * dZ.rowwise().sum();

        return std::make_tuple(dW, db, dZ);
    }

    // Updates layer parameters in place
    void update_params( Layer& layer, const Gradient& gradient, const double alpha) {
        layer.W = layer.W - alpha * gradient.dW;
        layer.b = layer.b - alpha * gradient.db;

    }

    void forward_prop(Eigen::MatrixXd& X) {
        // Forward Propagation
        Eigen::MatrixXd input = X;
        for (size_t j=0; j < layers.size() - 1; j++) {
            forward_prop_hidden_layer(layers[j], input);
            input = layers[j].A;
        }
        forward_prop_output(layers[layers.size() - 1], input);

    }

    void back_prop(const Eigen::MatrixXd& X, const Eigen::RowVectorXd& Y, double alpha) {
        // Back Propagation
        Gradient gradient;
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> backprop_data;
        backprop_data = back_prop_output(layers[layers.size()-2], layers[layers.size()-1], Y);
        gradient.dW = std::get<0>(backprop_data);
        gradient.db = std::get<1>(backprop_data);
        gradient.dZ = std::get<2>(backprop_data);
        update_params(layers[layers.size()-1], gradient, alpha);

        for (int j=layers.size()-2; j > 0 ; j--) {
            backprop_data = back_prop_hidden_layer(layers[j], layers[j+1], gradient, layers[j-1].A);
            gradient.dW = std::get<0>(backprop_data);
            gradient.db = std::get<1>(backprop_data);
            gradient.dZ = std::get<2>(backprop_data);
            update_params(layers[j], gradient, alpha);

        }

        backprop_data = back_prop_hidden_layer(layers[0], layers[1], gradient, X);
        gradient.dW = std::get<0>(backprop_data);
        gradient.db = std::get<1>(backprop_data);
        gradient.dZ = std::get<2>(backprop_data);
        update_params(layers[0], gradient, alpha);
    }

    Eigen::RowVectorXi get_predictions() {
        const Eigen::MatrixXd A = layers.back().A;
        int num_samples = A.cols();
        Eigen::RowVectorXi predictions(num_samples);

        for (int i = 0; i < num_samples; i++) {
            A.col(i).maxCoeff(&predictions(i));
        }

        return predictions;

    }

    double get_accuracy(Eigen::RowVectorXi predictions, Eigen::RowVectorXi Y) {
        const int m = Y.size();
        int correct_predictions = 0;

        for (int i=0; i < m; i++) {
            if (predictions(i) == Y(i)) {
                correct_predictions++;
            }
        }

        return static_cast<double>(correct_predictions) / static_cast<double>(m);
    }

};