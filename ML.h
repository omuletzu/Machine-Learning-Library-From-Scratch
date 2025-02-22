#ifndef ML_H
#define ML_H

#include <fstream>
#include <vector>
#include <stdexcept>
#include <random>
#include <string>
#include "Layer.h"

class ML {
public:
    ML() = default;

    std::string get_filename() const;
    bool load_model_file(std::string& filename);
    Matrix MSE_loss_function(Matrix output, std :: vector<std :: vector<double>> expected_output);
    Matrix MSE_loss_derived(Matrix output, std :: vector<std :: vector<double>> expected_output);
    Matrix cross_entropy_loss_function(Matrix output, std :: vector<std :: vector<double>> expected_output);
    Matrix cross_entropy_loss_derived(Matrix output, std :: vector<std :: vector<double>> expected_output);
    void forward(Matrix& input, void (*final_activation)(Matrix), void (*hidden_activation)(Matrix),
                 Matrix (*final_derivative)(Matrix), Matrix (*hidden_derivative)(Matrix),
                 Matrix (*final_cost)(Matrix, std :: vector<std :: vector<double>>),
                 Matrix (*final_cost_derivative)(Matrix, std :: vector<std :: vector<double>>),
                 std :: vector<std :: vector<double>> expected_output,
                 double learning_rate);
    void backward(const Matrix& activated_output, std :: vector<std :: vector<double>> expected_output,
                  Matrix (*final_derivative)(Matrix), Matrix (*hidden_derivative)(Matrix),
                  Matrix (*final_cost_derivative)(Matrix, std :: vector<std :: vector<double>>),
                  double learning_rate);

    std::string filename;
    std::ofstream fout;
    unsigned int layers_number = 0;
    std::vector<int> nodes_per_layer_list;
    std::vector<Layer> layers_list;
    std :: vector<Matrix> deactivated_value_per_layer;
    std :: vector<Matrix> activated_value_per_layers;
};

#endif

