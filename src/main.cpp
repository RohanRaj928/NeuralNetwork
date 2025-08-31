#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <future>
#include <thread>
#include <chrono>

#include <Eigen/Dense>
#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Widget.H>
#include <FL/fl_draw.H>
#include <sys/attr.h>

#include "neural_net.h"
#include "trainer.h"




/* Global Vars */
int dev_images_num = 100;
std::string training_filename = "dataset/mnist_test.csv";
double alpha = 0.1;

#if defined(_WIN32)
#include <windows.h>
std::filesystem::path getExecutablePath() {
	char buffer[MAX_PATH];
	GetModuleFileNameA(NULL, buffer, MAX_PATH);
	return std::filesystem::path(buffer).parent_path();
}
#elif defined(__linux__)
#include <unistd.h>
std::filesystem::path getExecutablePath() {
	char buffer[PATH_MAX];
	ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer)-1);
	if (len != -1) {
		buffer[len] = '\0';
		return std::filesystem::path(buffer).parent_path();
	}
	return std::filesystem::current_path(); // fallback
}
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
std::filesystem::path getExecutablePath() {
	char buffer[PATH_MAX];
	uint32_t size = sizeof(buffer);
	if (_NSGetExecutablePath(buffer, &size) == 0) {
		return std::filesystem::path(buffer).parent_path();
	}
	return std::filesystem::current_path(); // fallback
}
#else
std::filesystem::path getExecutablePath() {
	return std::filesystem::current_path(); // generic fallback
}
#endif

std::filesystem::path exe_dir = getExecutablePath();


/* File handling */


std::tuple<int, int> get_csv_size(const std::string& path) {
	std::filesystem::path real_path = exe_dir / path;
	std::ifstream file(real_path);
	std::string line;

	int rows = 0;
	int cols = 0;

	if (!file.is_open()) {
		throw std::runtime_error("Could not open file");
	}

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
	std::filesystem::path real_path = exe_dir / path;
    std::ifstream file(real_path);
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

/* UI */

class DrawingArea : public Fl_Widget {
	public:

	DrawingArea(int X_pos, int Y_pos, int grid_size, int cell_size, int brush_radius)
		: Fl_Widget(X_pos, Y_pos, cell_size * grid_size, cell_size * grid_size),
		  pixels(grid_size * grid_size, 0) {

		CELL_SIZE = cell_size;
		GRID_SIZE = grid_size;
		DRAW_SIZE = cell_size * cell_size;
		BRUSH_RADIUS = brush_radius;

	}

	void clear() {
		std::fill(pixels.begin(), pixels.end(), 0);
		redraw();
	}

	const std::vector<unsigned char>& get_pixels() const {
		return pixels;
	}

private:
	std::vector<unsigned char> pixels;
	int CELL_SIZE = 10;  // Each MNIST pixel is 10x10
	int GRID_SIZE = 28;  // 28x28 grid
	int DRAW_SIZE = CELL_SIZE * GRID_SIZE;
	int BRUSH_RADIUS = 2;

	int handle(int event) override {
		if (event == FL_DRAG || event == FL_PUSH) {
			int mx = Fl::event_x();
			int my = Fl::event_y();

			int gx = (mx - x()) / CELL_SIZE;
			int gy = (my - y()) / CELL_SIZE;

			if (mx >= x() && my >= y() && mx < x() + w() && my < y() + h()) {
				if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
					pixels[gy * GRID_SIZE + gx] = 255;
					redraw();
				}

				for (int dy = -BRUSH_RADIUS; dy <= BRUSH_RADIUS; ++dy) {
					for (int dx = -BRUSH_RADIUS; dx <= BRUSH_RADIUS; ++dx) {
						int nx = gx + dx;
						int ny = gy + dy;

						if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
							float dist = sqrtf(dx*dx + dy*dy);
							if (dist <= BRUSH_RADIUS) {
								float intensity = 1.0f - (dist / BRUSH_RADIUS);
								unsigned char brush_val = static_cast<unsigned char>(intensity * 255);
								int idx = ny * GRID_SIZE + nx;
								pixels[idx] = std::max(pixels[idx], brush_val);
							}
						}
					}
				}
				redraw();
				return 1;
			}
		}
		return 0;
	}

	void draw() override {
		for (int j = 0; j < GRID_SIZE; ++j) {
			for (int i = 0; i < GRID_SIZE; ++i) {
				unsigned char val = pixels[j * GRID_SIZE + i];
				fl_color(fl_rgb_color(val));  // White = 0, Black = 255
				fl_rectf(x() + i * CELL_SIZE, y() + j * CELL_SIZE, CELL_SIZE, CELL_SIZE);
			}
		}
		fl_color(FL_GRAY);
		fl_rect(x(), y(), w(), h());
	}
};

// Global pointers
Neural_net* g_neural_net = nullptr;
Trainer* g_trainer = nullptr;
Eigen::MatrixXd* X_test = nullptr;
Eigen::RowVectorXd* Y_test = nullptr;
Fl_Box* g_accuracy_label = nullptr;
Fl_Box* g_prediction_label = nullptr;
DrawingArea* g_drawing_area = nullptr;
std::string percent_str;  // hold text so pointer stays valid

void timer_callback(void*) {
	if (!g_trainer || !g_accuracy_label || !g_drawing_area || !X_test || !Y_test) return;

	// Calculate accuracy every 5 iterations;
	if (g_trainer->is_training()) {
		if (g_trainer->get_iterations() % 5 == 0) {
			double acc = g_trainer->get_accuracy(*X_test, *Y_test);
			// Format accuracy as percentage string
			std::ostringstream oss;
			oss << std::fixed << std::setprecision(1) << (acc * 100) << "%";
			percent_str = oss.str();

			// Use copy_label to safely update the label text
			g_accuracy_label->copy_label(percent_str.c_str());
			g_accuracy_label->redraw();
		}
	}


	// Make prediction when model not training
	if (!g_trainer->is_training()) {
	 	Eigen::VectorXd X_input(784);
	 	int drawing_area_size = g_drawing_area->get_pixels().size();
	 	for (int i = 0; i < drawing_area_size; ++i) {
	 		X_input(i) = g_drawing_area->get_pixels()[i];
	 	}
	 	g_prediction_label->copy_label(std::to_string(g_neural_net->make_prediction(X_input)).c_str());
	 	g_prediction_label->redraw();
	}

	// Reschedule timer on main thread
	Fl::repeat_timeout(0.1, timer_callback);
}

int main(int argc, char **argv) {
	// FL setup
	constexpr int window_width = 420;
	constexpr int window_height = 320;


	// Dataset
	std::cout << "Loading Dataset" << std::endl;

    const std::tuple<int, int> result = get_csv_size(training_filename);	// Gets dimensions of the csv file
    const int rows = std::get<0>(result);
    const int cols = std::get<1>(result);

	Eigen::MatrixXd data = load_csv(training_filename, rows, cols); // Loads the csv file

	std::cout << "Manipulating Dataset" << std::endl;
	data.transposeInPlace();											// Transpose so that the files are 'vertical'

	// Splitting dataset

	Eigen::MatrixXd data_dev = data.leftCols(dev_images_num); 			// Use the first n images to use for development
	Eigen::RowVectorXd Y_dev = data_dev.row(0);								// First row of data: label showing the number
	Eigen::MatrixXd X_dev = data_dev.middleRows(1, data_dev.rows() - 1 );	// Actual image data
	X_dev = X_dev / 255.0;

	Eigen::MatrixXd data_train = data.rightCols(data.cols() - dev_images_num);	// Images used for training
	Eigen::RowVectorXd Y_train = data_train.row(0);
	Eigen::MatrixXd X_train = data_train.middleRows(1, data_train.rows() - 1);
	X_train = X_train / 255.0;
	size_t m_train = Y_train.size();

	X_test = &X_dev;
	Y_test = &Y_dev;

	Neural_net neural_net({
		{784,10},
		{10, 10}
	}, m_train);
	g_neural_net = &neural_net;

	Trainer trainer(neural_net);
	trainer.set_accuracy_callback(X_dev, Y_dev, 5);
	g_trainer = &trainer;

	// UI

	Fl_Window* window = new Fl_Window(window_width, window_height, "MNIST Drawer");

	// Prediction label on the left
	Fl_Box* prediction_box = new Fl_Box(20, 20, 80, 40, "Prediction:\n?");
	prediction_box->box(FL_DOWN_BOX);
	prediction_box->align(FL_ALIGN_INSIDE | FL_ALIGN_WRAP);
	g_prediction_label = prediction_box;

	// Drawing area on the right
	DrawingArea* canvas = new DrawingArea(120, 20, 28, 10, 2);
	g_drawing_area = canvas;

	// Clear button below
	Fl_Button* clear_button = new Fl_Button(20, 65, 80, 30, "Clear");
	clear_button->callback([](Fl_Widget*, void* data) {
		DrawingArea* canvas = static_cast<DrawingArea *>(data);
		canvas->clear();
	}, canvas);

	using CallbackDataType = std::tuple<Trainer&, const Eigen::MatrixXd&, const Eigen::RowVectorXd&, double&>;
	auto callback_data = new CallbackDataType{trainer, X_train, Y_train, alpha};
	Fl_Button* train_button = new Fl_Button(20, 140, 80, 30, "Train");
	train_button->callback([](Fl_Widget* w, void* userdata) {
		using CallbackDataType = std::tuple<Trainer&, Eigen::MatrixXd&, Eigen::RowVectorXd&, double&>;

		auto* data_ptr = static_cast<CallbackDataType*>(userdata);

		Trainer& trainer = std::get<0>(*data_ptr);
		if (trainer.is_training()) {
			trainer.stop_training();
			w->label("Train");
		}
		else {
			Eigen::MatrixXd& X_train = std::get<1>(*data_ptr);
			Eigen::RowVectorXd& Y_train = std::get<2>(*data_ptr);
			double& alpha = std::get<3>(*data_ptr);
			trainer.start_training(X_train, Y_train, alpha);

			w->label("Stop");
		}


	}, callback_data);


	Fl_Box* label = new Fl_Box(20, 175, 80, 30, "0%");
	label->labelsize(10);
	g_accuracy_label = label;


	window->end();
	window->show();

	Fl::add_timeout(0.1, timer_callback);



	const int ret = Fl::run();
	trainer.stop_training();
	trainer.wait();


	return ret;

}
