#include "Layer.h"
#include "ML.h"
#include <iostream>

bool Layer :: set_matrixes(std :: ifstream& fin) {
    std :: vector<std::vector<double>> weight_final_matrix = std :: vector<std :: vector<double>>();
    std :: vector<std::vector<double>> bias_final_matrix = std :: vector<std :: vector<double>>();

    if(this -> current_layer_nodes <= 0){
        return false;
    }

    for(int i = 0; i < current_layer_nodes; i++) {

        if(this -> prev_layer_nodes <= 0){
            return false;
        }

        std :: vector<double> weight_matrix_line = std :: vector<double>(prev_layer_nodes);

        for(int j = 0; j < prev_layer_nodes; j++) {
            fin >> weight_matrix_line[j];
        }

        weight_final_matrix.push_back(weight_matrix_line);

        std :: vector<double> bias_one_line = std :: vector<double>(1);

        fin >> bias_one_line[0];

        bias_final_matrix.push_back(bias_one_line);
    }

    this -> matrix_weight = Matrix(weight_final_matrix);
    this -> matrix_bias = Matrix(bias_final_matrix);

    return true;
}

Layer::Layer() {}
