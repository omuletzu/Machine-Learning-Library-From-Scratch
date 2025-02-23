#ifndef MATRIX_M
#define MATRIX_M

#include <vector>
#include <stdexcept>
#include <iostream>

class Matrix {

public:
    Matrix();

    Matrix(int row, int col) : row(row), col(col) {
        this->matrix = std::vector<std::vector<double>>(row, std::vector<double>(col, 0));
    };

    Matrix(int row, int col, double init_value) : row(row), col(col) {
        this->matrix = std::vector<std::vector<double>>(row, std::vector<double>(col, init_value));
    };

    Matrix(std::vector<std::vector<double>> init_data) {
        if(init_data.empty()){
            throw std::invalid_argument("Invalid row");
        }

        for(auto& elem : init_data){
            if(elem.empty()){
                throw std::invalid_argument("Invalid col");
            }
        }

        this->row = init_data.size();
        this->col = init_data[0].size();
        this->matrix = init_data;
    }

    std::vector<std::vector<double>> get_matrix();
    std::pair<size_t, size_t> get_row_col();
    double at(int i, int j);
    void set(int i, int j, double value);
    static Matrix add_matrix(const Matrix& a, const Matrix& b);
    static Matrix sub_matrix(const Matrix& a, const Matrix& b);
    static Matrix mul_matrix(const Matrix& a, const Matrix& b);
    static Matrix mul_simple_matrix(const Matrix& a, const Matrix& b);
    static Matrix div_matrix(const Matrix& a, const Matrix& b);
    static Matrix transpose(const Matrix& a);
    static void reLU_activation(Matrix& value);
    static Matrix reLU_derivative(Matrix value);
    static void sigmoid_activation(Matrix& value);
    static Matrix sigmoid_derivative(Matrix value);
    static void softmax_activation(Matrix& value);
    static Matrix softmax_derivative(Matrix value);

private:
    unsigned int row;
    unsigned int col;
    std::vector<std::vector<double>> matrix;
};

#endif
