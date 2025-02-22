#include "Matrix.h"
#include "math.h"

bool check_bound(int i, int j, unsigned int row, unsigned int col){
    return i < row && i >= 0 && j < col && j >= 0;
}

double Matrix :: at(int i, int j) {
    if(check_bound(i, j, this->row, this->col)){
        return this->matrix[i][j];
    }

    return 0.0;
}

void Matrix :: set(int i, int j, double value) {
    if(check_bound(i, j, this->row, this->col)){
        this->matrix[i][j] = value;
    }
}

std::vector<std::vector<double>> Matrix :: get_matrix() {
    return this->matrix;
}

std::vector<double> Matrix :: get_row_col() const {
    std::vector<double> row_col = std::vector<double>(2);
    row_col[0] = this->row;
    row_col[1] = this->col;

    return row_col;
}

Matrix Matrix :: add_matrix(const Matrix& a, const Matrix& b) {
    if(a.matrix.size() != b.matrix.size()){
        return {0, 0};
    }

    if(a.matrix[0].size() != b.matrix[0].size()){
        return {0, 0};
    }

    Matrix result = a;

    for(int i = 0; i < b.matrix.size(); i++){
        for(int j = 0; j < result.matrix.size(); j++){
            result.matrix[i][j] += b.matrix[i][j];
        }
    }

    return result;
}

Matrix Matrix :: sub_matrix(const Matrix& a, const Matrix& b) {
    if(a.matrix.size() != b.matrix.size()){
        return {0, 0};
    }

    if(a.matrix[0].size() != b.matrix[0].size()){
        return {0, 0};
    }

    Matrix result = a;

    for(int i = 0; i < b.matrix.size(); i++){
        for(int j = 0; j < result.matrix.size(); j++){
            result.matrix[i][j] -= b.matrix[i][j];
        }
    }

    return result;
}

Matrix Matrix :: mul_matrix(const Matrix& a, const Matrix& b) {
    if(a.matrix[0].size() != b.matrix.size()){
        return {0, 0};
    }

    Matrix result(a.matrix.size(), b.matrix[0].size());

    for(int i = 0; i < a.matrix.size(); i++){
        for(int j = 0; j < b.matrix[0].size(); j++){
            for(int k = 0; k < b.matrix.size(); k++){
                result.matrix[i][j] += a.matrix[i][k] * b.matrix[k][j];
            }
        }
    }

    return result;
}

Matrix Matrix :: div_matrix(const Matrix& a, const Matrix& b) {
    if(a.matrix.size() != b.matrix.size()){
        return {0, 0};
    }

    if(a.matrix[0].size() != b.matrix[0].size()){
        return {0, 0};
    }

    Matrix result = a;

    for(int i = 0; i < b.matrix.size(); i++){
        for(int j = 0; j < result.matrix.size(); j++){
            result.matrix[i][j] /= b.matrix[i][j];
        }
    }

    return result;
}

Matrix Matrix :: mul_simple_matrix(const Matrix &a, const Matrix &b) {
    if(a.matrix.size() != b.matrix.size()){
        return {0, 0};
    }

    if(a.matrix[0].size() != b.matrix[0].size()){
        return {0, 0};
    }

    Matrix result = a;

    for(int i = 0; i < b.matrix.size(); i++){
        for(int j = 0; j < result.matrix.size(); j++){
            result.matrix[i][j] *= b.matrix[i][j];
        }
    }

    return result;
}

Matrix Matrix::transpose(const Matrix &a) {
    Matrix result(a.matrix[0].size(), a.matrix.size());

    for(int i = 0; i < result.matrix.size(); i++){
        for(int j = 0; j < result.matrix[i].size(); j++){
            result.matrix[i][j] = a.matrix[j][i];
        }
    }

    return result;
}

void Matrix::reLU_activation(Matrix& value) {
    for(auto & i : value.matrix){
        for(double & j : i){
            j = (j > 0 ? j : 0);
        }
    }
}

Matrix Matrix :: reLU_derivative(Matrix value) {
    for(auto & i : value.matrix){
        for(double & j : i){
            j = (j > 0 ? 1 : 0);
        }
    }

    return value;
}

void Matrix::sigmoid_activation(Matrix& value) {
    for(auto & i : value.matrix){
        for(double & j : i){
            j = 1 / (1 + exp(-1 * j));
        }
    }
}

Matrix Matrix :: sigmoid_derivative(Matrix value) {
    for(auto & i : value.matrix){
        for(double & j : i){
            double sigmoid = 1 / (1 + exp(-1 * j));
            j = sigmoid * (1 - sigmoid);
        }
    }

    return value;
}

void Matrix::softmax_activation(Matrix& value) {
    for(int i = 0; i < value.matrix[0].size(); i++){
        double exp_sum = 0.0;

        for(auto & j : value.matrix){
            exp_sum += j[i];
        }

        for(auto & j : value.matrix){
            j[i] = exp(j[i]) / exp_sum;
        }
    }
}

Matrix Matrix :: softmax_derivative(Matrix value) {
    for(int i = 0; i < value.matrix[0].size(); i++){
        double exp_sum = 0.0;

        for(auto & j : value.matrix){
            exp_sum += j[i];
        }

        for(auto & j : value.matrix){
            double softmax = exp(j[i]) / exp_sum;
            j[i] = softmax * (1 - softmax);
        }
    }

    return value;
}

Matrix::Matrix() {}
