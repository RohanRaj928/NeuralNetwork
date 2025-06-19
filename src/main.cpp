#include <iostream>
#include <fstream>
#include <tuple>

#include <Eigen/Dense>

// Global Vars
int dev_images_num = 100;
std::string training_filename = "dataset/mnist_test.csv";



std::tuple<int, int> get_csv_size(const std::string& path) {
    std::ifstream file(path);
    std::string line;

    int rows = 0;
    int cols = 0;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    // Read header line to get number of columns
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            ++cols;
        }
    }

    // Count remaining lines (rows)
    while (std::getline(file, line)) {
        ++rows;
    }

    file.close();

    return std::make_tuple(rows, cols);
}

Eigen::MatrixXd load_csv(const std::string& path, int rows, int cols) {
    std::ifstream file(path);
    std::string line;
    std::vector<double> values;
    int row_count = 0;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line) && row_count < rows) {
        std::stringstream ss(line);
        std::string cell;
        int col_count = 0;
        while (std::getline(ss, cell, ',') && col_count < cols) {
            values.push_back(std::stod(cell));
            col_count++;
        }
        row_count++;
    }

    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        values.data(), rows, cols);
}

struct Layer {				// Struct represents a layer
	Eigen::MatrixXd W;	// Weights
	Eigen::VectorXd b;	// Biases
	Eigen::VectorXd Z;	// Pre-Activation data
	Eigen::VectorXd A; 	// Activation data

	Layer(int input, int output) {
		W = Eigen::MatrixXd::Random(output, input) * 0.5f;
		b = Eigen::VectorXd::Random(output) * 0.5f;
		Z = Eigen::VectorXd::Zero(output);
		A = Eigen::VectorXd::Zero(output);
	}

};

struct Gradient {
	Eigen::MatrixXd dZ;
	Eigen::MatrixXd dW;
	Eigen::MatrixXd db;

};

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

Eigen::MatrixXd ReLU(const Eigen::MatrixXd& Z) {
    return Z.array().max(0.0);
}

Eigen::MatrixXd deriv_ReLU(const Eigen::MatrixXd& Z) {
	return (Z.array() > 0).cast<double>();

}

Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z) {
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

    // divide each column by sum of its exponentials
    Eigen::MatrixXd A = exp_Z;
    for (int i = 0; i < exp_Z.cols(); ++i) {
        A.col(i).array() /= sum_exp(i);
    }

    return A;
}

Eigen::MatrixXd forward_prop_hidden(const Layer& layer, const Eigen::MatrixXd& X) {
    Eigen::MatrixXd Z = (layer.W * X);
    Z.colwise() += layer.b.col(0);
    return ReLU(Z);

}

Eigen::MatrixXd forward_prop_output(const Layer& layer, const Eigen::MatrixXd& X){
    Eigen::MatrixXd Z = (layer.W * X);
    Z.colwise() += layer.b.col(0);
    return softmax(Z);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
back_prop_output(const Layer& layer1, const Layer& layer2,
				 const Eigen::MatrixXd& X, const Eigen::RowVectorXd& Y){
	int m = Y.size();
	Eigen::MatrixXd dZ = layer2.A - one_hot(Y);
	Eigen::MatrixXd dW = (1.0 / m) * (dZ * layer1.A.transpose());
	Eigen::MatrixXd db = (1.0 / m) * dZ.rowwise().sum();

	return std::make_tuple(dW, db, dZ);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
back_prop_hidden(const Layer& layer1, const Layer& layer2,
				 const Gradient& gradient,
				 const Eigen::MatrixXd& X, const Eigen::RowVectorXd& Y) {
	int m = Y.size();
	Eigen::MatrixXd dZ = (layer2.W.transpose() * gradient.dZ).array() * deriv_ReLU(layer1.Z).array();
	Eigen::MatrixXd dW = (1.0 / m) * (dZ * X.transpose());
	Eigen::MatrixXd db = (1.0 / m) * dZ.rowwise().sum();

	return std::make_tuple(dW, db);

}




int main() {
	/* Data Set */
	std::cout << "Loading Dataset" << std::endl;

    std::tuple<int, int> result = get_csv_size(training_filename);	// Gets dimensions of the csv file
    int rows = std::get<0>(result);
    int cols = std::get<1>(result);

	Eigen::MatrixXd data = load_csv(training_filename, rows, cols); // Loads the csv file

	std::cout << "Manipulating Dataset" << std::endl;
	data.transposeInPlace();											// Transpose so that the files are 'vertical'

	/* Splitting Dataset*/

	Eigen::MatrixXd data_dev = data.leftCols(dev_images_num); 			// Use the first n images to use for development
	Eigen::RowVectorXd Y_dev = data_dev.row(0);								// First row of data: label showing the number
	Eigen::MatrixXd X_dev = data_dev.middleRows(1, data_dev.rows() - 1 );	// Actual image dataa



	Eigen::MatrixXd data_train = data.rightCols(data.cols() - dev_images_num);
	Eigen::RowVectorXd Y_train = data_train.row(0);
	Eigen::MatrixXd X_train = data_train.middleRows(1, data_train.rows() - 1);



}
