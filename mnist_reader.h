#ifndef ML_LIB_MNIST_READER_H
#define ML_LIB_MNIST_READER_H

#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

std :: vector<std :: vector<double>> read_dataset(const std :: string& filename) {
    std :: ifstream file = std :: ifstream(filename, std :: ios :: binary);

    if(file.is_open()){

        int magic = 0;
        int number_of_images = 0;
        int row_number = 0;
        int col_number = 0;

        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&number_of_images), 4);
        file.read(reinterpret_cast<char*>(&row_number), 4);
        file.read(reinterpret_cast<char*>(&col_number), 4);

        magic = __builtin_bswap32(magic);

        if(magic != 0x00000803) {
            throw std :: invalid_argument("Wrong mnist format");
        }

        number_of_images = __builtin_bswap32(number_of_images);
        row_number = __builtin_bswap32(row_number);
        col_number = __builtin_bswap32(col_number);

        std :: vector<std :: vector<double>> mnist_images_vector = std :: vector<std :: vector<double>>();

        for(int i = 0; i < number_of_images; i++) {

            std :: vector<uint8_t> mnist_image(row_number * col_number);
            file.read(reinterpret_cast<char*>(mnist_image.data()), row_number * col_number);

            std :: vector<double> mnist_normalized_image(mnist_image.size());

            std :: transform(mnist_image.begin(), mnist_image.end(), mnist_normalized_image.begin(),
                             [](uint8_t pixel) { return pixel / 255.0; });

            mnist_images_vector.emplace_back(mnist_normalized_image);
        }

        return mnist_images_vector;
    }

    return std :: vector<std :: vector<double>>(0);
}

std :: vector<uint8_t> read_dataset_labels(const std :: string& filename) {
    std :: ifstream file = std :: ifstream(filename, std :: ios :: binary);

    if(file.is_open()){

        int magic = 0;
        int number_of_labels = 0;

        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&number_of_labels), 4);

        magic = __builtin_bswap32(magic);

        if(magic != 0x00000801) {
            throw std :: invalid_argument("Wrong mnist format");
        }

        number_of_labels = __builtin_bswap32(number_of_labels);

        std :: vector<uint8_t> mnist_labels_vector = std :: vector<uint8_t>(number_of_labels);

        file.read(reinterpret_cast<char*> (mnist_labels_vector.data()), number_of_labels);

        return mnist_labels_vector;
    }

    return std :: vector<uint8_t>(0);
}

#endif //ML_LIB_MNIST_READER_H
