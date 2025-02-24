#include "ML.h"
#include <stdexcept>
#include "Matrix.h"
#include <iostream>
#include <utility>

std::string ML :: get_filename() const {
    return this -> filename;
}

bool ML :: load_model_file(std::string& filename) {

    if(filename.empty()){
        throw std::invalid_argument("Invalid file");
    };

    this -> filename = filename;

    std :: ifstream fin = std :: ifstream(this -> filename);

    if(fin.is_open()){
        fin >> this -> layers_number;

        this -> nodes_per_layer_list = std::vector<int>(layers_number);
        this -> layers_list = std :: vector<Layer>();

        for(int i = 0; i < layers_number; i++){
            fin >> this -> nodes_per_layer_list[i];
        }

        this -> layers_list.emplace_back();

        for(int i = 1; i < this -> layers_number; i++){
            this -> layers_list.emplace_back(this -> nodes_per_layer_list[i], this -> nodes_per_layer_list[i - 1], fin);
        }

        this -> deactivated_value_per_layer = std :: vector<Matrix>(this -> layers_number);
        this -> activated_value_per_layers = std :: vector<Matrix>(this -> layers_number);

        fin.close();

        return true;
    }

    fin.close();

    return false;
}

void ML :: update_model_file() {
    std :: ofstream log_weight_bias = std :: ofstream(this -> filename);

    log_weight_bias << this -> layers_number << "\n";

    for(int i = 0; i < this -> layers_number; i++){
        log_weight_bias << this -> nodes_per_layer_list[i] << "\n";
    }

    for(int i = 1; i < this -> layers_list.size(); i++) {
        Matrix weight_matrix = this -> layers_list[i].matrix_weight.get_matrix();
        Matrix bias_matrix = this -> layers_list[i].matrix_bias.get_matrix();

        for(int j = 0; j < this -> layers_list[i].current_layer_nodes; j++){
            for(int k = 0; k < this -> layers_list[i].prev_layer_nodes; k++){
                log_weight_bias << weight_matrix.at(j, k) << " ";
            }

            log_weight_bias << bias_matrix.at(j, 0) << "\n";
        }
    }

    log_weight_bias.close();
}

Matrix ML :: MSE_loss_function(Matrix output, std :: vector<std :: vector<double>> expected_output) {
    std :: vector<std :: vector<double>> output_matrix = output.get_matrix();

    Matrix cost(1, output_matrix[0].size());

    for(int i = 0; i < output_matrix[0].size(); i++){
        double final_sum = 0.0;

        for(int j = 0; j < output_matrix.size(); j++){
            final_sum += (output_matrix[j][i] - expected_output[j][i]) * (output_matrix[j][i] - expected_output[j][i]);
        }

        cost.set(0, i, final_sum * 0.5);
    }

    return cost;
}

Matrix ML :: MSE_loss_derived(Matrix output, std :: vector<std :: vector<double>> expected_output) {
    std :: vector<std :: vector<double>> output_matrix = output.get_matrix();

    Matrix derived_matrix(output_matrix.size(), output_matrix[0].size());

    for(int i = 0; i < output_matrix.size(); i++){
        for(int j = 0; j < output_matrix[i].size(); j++){
            derived_matrix.set(i, j, output_matrix[i][j] - expected_output[i][j]);
        }
    }

    return derived_matrix;
}

Matrix ML::cross_entropy_loss_function_with_softmax(Matrix output, std::vector<std::vector<double>> expected_output) {
    std :: vector<std :: vector<double>> output_matrix = output.get_matrix();

    Matrix cost(1, output_matrix[0].size());

    for(int i = 0; i < output_matrix[0].size(); i++){
        double final_sum = 0.0;

        for(int j = 0; j < output_matrix.size(); j++){
            if(output_matrix[j][i] > 0){
                final_sum += expected_output[j][i] * log(std :: max(output_matrix[j][i], 1e-9));
            }
        }

        cost.set(0, i, -1 * final_sum);
    }

    return cost;
}

Matrix ML::cross_entropy_loss_with_softmax_derived(Matrix output, std::vector<std::vector<double>> expected_output) {
    std :: vector<std :: vector<double>> output_matrix = output.get_matrix();

    Matrix derived_matrix(output_matrix.size(), output_matrix[0].size());

    for(int i = 0; i < output_matrix.size(); i++){
        for(int j = 0; j < output_matrix[i].size(); j++){
            derived_matrix.set(i, j, output_matrix[i][j] - expected_output[i][j]);
        }
    }

    return derived_matrix;
}

Matrix ML :: forward(Matrix& input,
                   void (*final_activation)(Matrix&),
                   void (*hidden_activation)(Matrix&),
                   Matrix (*final_derivative)(Matrix),
                   Matrix (*hidden_derivative)(Matrix),
                   Matrix (*final_cost)(Matrix, std :: vector<std :: vector<double>>),
                   Matrix (*final_cost_derivative)(Matrix, std :: vector<std :: vector<double>>),
                   std :: vector<std :: vector<double>> expected_output,
                   double learning_rate,
                   bool enable_backward) {

    std :: pair<int, int> row_col = input.get_row_col();
    int row = row_col.first;

    if(row != this -> nodes_per_layer_list[0]){
        throw std :: invalid_argument("Invalid input");
    }

    if(expected_output.size() != nodes_per_layer_list[nodes_per_layer_list.size() - 1]){
        throw std :: invalid_argument("Invalid expected output");
    }

    this -> deactivated_value_per_layer[0] = input;
    this -> activated_value_per_layers[0] = input;

    Matrix partial_result = input;

    for(int i = 1; i < this -> layers_list.size(); i++) {
        partial_result = Matrix :: add_broadcast_matrix(Matrix :: mul_matrix(this -> layers_list[i].matrix_weight, partial_result), this -> layers_list[i].matrix_bias);

        this -> deactivated_value_per_layer[i] = partial_result;

        if(i < this -> layers_list.size() - 1) {
            hidden_activation(partial_result);
        }
        else{
            final_activation(partial_result);
        }

        this -> activated_value_per_layers[i] = partial_result;
    }

    if(enable_backward) {
        backward(partial_result, expected_output, final_derivative, hidden_derivative, final_cost_derivative, learning_rate);
    }

    return final_cost(partial_result, expected_output);
}

void correct_weight_bias(Layer& layer, Matrix gradient_values, Matrix& activation_values, double learning_rate) {

    Matrix weight_gradient_matrix = Matrix :: mul_matrix(gradient_values, Matrix :: transpose(activation_values));

    std :: vector<std :: vector<double>> weight_matrix_get = layer.matrix_weight.get_matrix();
    std :: vector<std :: vector<double>> weight_gradient_matrix_get = weight_gradient_matrix.get_matrix();

    std :: vector<std :: vector<double>> divide_multiple_active_inputs_matrix = std :: vector<std :: vector<double>>(
            weight_gradient_matrix_get.size(), std :: vector<double>(weight_gradient_matrix_get[0].size(), 1 / gradient_values.get_matrix()[0].size())
            );
    Matrix divide_multiple_active_inputs = Matrix(divide_multiple_active_inputs_matrix);

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

void ML :: backward(const Matrix& activated_output,
                    std :: vector<std :: vector<double>> expected_output,
                    Matrix (*final_derivative)(Matrix),
                    Matrix (*hidden_derivative)(Matrix),
                    Matrix (*final_cost_derivative)(Matrix, std :: vector<std :: vector<double>>),
                    double learning_rate) {

    Matrix current_layer_gradient = Matrix :: mul_simple_matrix(final_cost_derivative(activated_output, expected_output), final_derivative(deactivated_value_per_layer[deactivated_value_per_layer.size() - 1]));

    correct_weight_bias(layers_list[layers_list.size() - 1], current_layer_gradient, activated_value_per_layers[activated_value_per_layers.size() - 2], learning_rate);

    for(int i = layers_list.size() - 2; i > 0; i--) {
        current_layer_gradient = Matrix :: mul_simple_matrix(Matrix :: mul_matrix(Matrix :: transpose(layers_list[i + 1].matrix_weight), current_layer_gradient), hidden_derivative(deactivated_value_per_layer[i]));
        correct_weight_bias(layers_list[i], current_layer_gradient, activated_value_per_layers[i - 1], learning_rate);
    }
}
