#ifndef LAYER_H
#define LAYER_H

#include "Node.h"
#include "Matrix.h"

class Layer {
public:
    Layer(int current_layer_nodes, int prev_layer_nodes, std::ifstream& fin) :
        current_layer_nodes(current_layer_nodes),
        prev_layer_nodes(prev_layer_nodes),
        matrix_weight(Matrix(0, 0)),
        matrix_bias(Matrix(0, 0)) {
        set_matrixes(fin);
    };

    bool set_matrixes(std :: ifstream& fin);
    unsigned int current_layer_nodes;
    unsigned int prev_layer_nodes;
    Matrix matrix_weight;
    Matrix matrix_bias;
};

#endif