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

    void set_filename(const std::string& filename);
    std::string get_filename() const;
    bool load_model_file(const std::string& filename);
    Matrix MSE_loss_function(Matrix output, std :: vector<std :: vector<double>> expected_output);
    Matrix MSE_loss_derived(Matrix output, std :: vector<std :: vector<double>> expected_output);
    Matrix cross_entropy_loss_function(Matrix output);
    void forward(const Matrix& input, void (*final_activation)(Matrix), std :: vector<std :: vector<double>> expected_output, double learning_rate);
    void backward(Matrix activated_output, std :: vector<std :: vector<double>> expected_output, double learning_rate);

    std::string filename;
    std::ofstream fout;
    unsigned int layers_number = 0;
    std::vector<int> nodes_per_layer_list;
    std::vector<Layer> layers_list;
    std :: vector<Matrix> deactivated_value_per_layer;
    std :: vector<Matrix> activated_value_per_layers;
};

#endif

