#include "ML.h"
#include <stdexcept>
#include "Matrix.h"
#include <math.h>

std::string ML :: get_filename() const {
    return this -> filename;
}

bool ML :: load_model_file(const std::string& filename) {

    if(filename.empty()){
        throw std::invalid_argument("Invalid file");
    };

    this -> filename = filename;

    std :: ifstream fin = std :: ifstream(this -> filename);

    if(fin.is_open()){
        fin >> layers_number;

        this -> nodes_per_layer_list = std::vector<int>(layers_number);
        this -> layers_list = std :: vector<Layer>();

        for(int i = 0; i < layers_number; i++){
            fin >> this -> nodes_per_layer_list[i];
        }

        for(int i = 1; i < this -> layers_number; i++){
            this -> layers_list.emplace_back(this -> nodes_per_layer_list[i], this -> nodes_per_layer_list[i - 1], fin);
        }

        this -> deactivated_value_per_layer = std :: vector<Matrix>(this -> layers_number - 1);
        this -> activated_value_per_layers = std :: vector<Matrix>(this -> layers_number - 1);

        return true;
    }

    return false;
}

Matrix ML :: MSE_loss_function(Matrix output, std :: vector<std :: vector<double>> expected_output) {
    std :: vector<std :: vector<double>> output_matrix = output.get_matrix();

    Matrix cost(1, output_matrix[0].size());

    for(int i = 0; i < output_matrix[0].size(); i++){
        for(int j = 0; j < output_matrix.size(); j++){
            double partial_sum = cost.at(0, i);
            cost.set(0, i, partial_sum + (output_matrix[j][i] - expected_output[j][i]) * (output_matrix[j][i] - expected_output[j][i]));
        }

        cost.set(0, i, cost.at(0, i) * 0.5);
    }

    return cost;
}

Matrix ML :: MSE_loss_derived(Matrix output, std :: vector<std :: vector<double>> expected_output) {
    std :: vector<std :: vector<double>> output_matrix = output.get_matrix();

    Matrix derived_matrix(output_matrix.size(), output_matrix[0].size());

    for(int i = 0; i < output_matrix.size(); i++){
        for(int j = 0; j < output_matrix[i].size(); i++){
            derived_matrix.set(i, j, output_matrix[i][j] - expected_output[i][j]);
        }
    }

    return derived_matrix;
}

void ML :: forward(const Matrix& input, void (*final_activation)(Matrix), std :: vector<std :: vector<double>> expected_output, double learning_rate){
    std::vector<double> row_col = input.get_row_col();
    double row = row_col[0];
    double col = row_col[1];

    if(row != this -> nodes_per_layer_list[0]){
        throw std :: invalid_argument("Invalid input");
    }

    Matrix partial_result = input;

    for(int i = 0; i < this -> layers_list.size(); i++) {
        partial_result = Matrix :: add_matrix(Matrix :: mul_matrix(this -> layers_list[i].matrix_weight, partial_result), this -> layers_list[i].matrix_bias);

        this -> deactivated_value_per_layer[i] = partial_result;

        if(i < this -> layers_list.size() - 1) {
            Matrix ::reLU_activation(partial_result);
        }
        else{
            final_activation(partial_result);
        }

        this -> activated_value_per_layers[i] = partial_result;
    }

    backward(partial_result, expected_output, learning_rate);
}

void correct_weight_bias(Layer& layer, Matrix gradient_values, Matrix& activation_values, double learning_rate) {
    Matrix weight_gradient_matrix = Matrix :: mul_matrix(gradient_values, Matrix :: transpose(activation_values));

    std :: vector<std :: vector<double>> weight_matrix_get = layer.matrix_weight.get_matrix();
    std :: vector<std :: vector<double>> weight_gradient_matrix_get = weight_gradient_matrix.get_matrix();

    for(int i = 0; i < weight_matrix_get.size(); i++) {
        for(int j = 0; j < weight_matrix_get[i].size(); j++) {
            layer.matrix_weight.set(i, j, layer.matrix_weight.at(i, j) - learning_rate * weight_gradient_matrix.at(i, j));
        }
    }

    std :: vector<std :: vector<double>> bias_matrix_get = layer.matrix_bias.get_matrix();
    std :: vector<std :: vector<double>> bias_gradient_matrix_get = gradient_values.get_matrix();

    for(int i = 0; i < bias_matrix_get.size(); i++) {
        layer.matrix_bias.set(i, 0, layer.matrix_bias.at(i, 0) - learning_rate * bias_gradient_matrix_get[i][0]);
    }
}

void ML :: backward(Matrix activated_output, std :: vector<std :: vector<double>> expected_output, double learning_rate) {
    Matrix current_layer_gradient = Matrix :: mul_simple_matrix(MSE_loss_derived(activated_output, expected_output), Matrix :: softmax_derivative(activated_output));

    correct_weight_bias(layers_list[layers_list.size() - 1], current_layer_gradient, activated_value_per_layers[activated_value_per_layers.size() - 2], learning_rate);

    for(int i = layers_list.size() - 2; i >= 0; i--) {
        current_layer_gradient = Matrix :: mul_simple_matrix(Matrix :: mul_matrix(Matrix :: transpose(layers_list[i + 1].matrix_weight), current_layer_gradient), Matrix :: reLU_derivative(deactivated_value_per_layer[i]));
        correct_weight_bias(layers_list[i], current_layer_gradient, activated_value_per_layers[i - 1], learning_rate);
    }
}