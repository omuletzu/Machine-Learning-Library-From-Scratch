#include <iostream>
#include <string>
#include "ML.h"
#include "Matrix.h"
#include "mnist_reader.h"

int main() {

    int state = 0;

    ML model = ML();

    while(true){
        if(state == 0){
            std :: cout << "Choose an option\n";
            std :: cout << "\t 1. Create new model file\n";
            std :: cout << "\t 2. Load model file\n";
            std :: cout << "\t 3. Train model\n";
            std :: cout << "\t 4. Test model\n";
            std :: cin >> state;
        }

        if(state == 1){
            unsigned int layers_number;
            unsigned int nodes_per_layer;

            std::string filename;
            std :: cout << "Filename:\n";
            std :: cin >> filename;
            std::ofstream fout = std::ofstream(filename);

            std :: cout << "Number of layers:\n";
            std :: cin >> layers_number;

            fout << layers_number << "\n";

            std::vector<unsigned int> nodes_per_layer_list(layers_number);

            for(int i = 0; i < layers_number; i++){
                std :: cout << "Number of nodes for layer - " << i << " :\n";
                std :: cin >> nodes_per_layer;
                nodes_per_layer_list[i] = nodes_per_layer;
                fout << nodes_per_layer << "\n";
            }

            std::random_device rd;
            std::default_random_engine generator(rd());
            std::uniform_real_distribution<double> distribution(-0.05, 0.05);

            for(int i = 1; i < layers_number; i++){
                for(int j = 0; j < nodes_per_layer_list[i]; j++){
                    for(int k = 0; k < nodes_per_layer_list[i - 1]; k++){
                        fout << distribution(generator) << " ";
                    }

                    fout << distribution(generator) << "\n";
                }
            }

            fout.close();

            state = 0;
        }

        if(state == 2){
            std :: string filename;
            std :: cout << "Filename:\n";
            std :: cin >> filename;

            if(model.load_model_file(filename)){
                std :: cout << "Loaded model - " + filename + " -\n";
            }
            else{
                std :: cout << "Failed loading - " + filename + " -\n";
            }

            state = 0;
        }

        if(state == 3){
            std :: vector<std :: vector<double>> train_dataset = read_dataset("mnist/train-images-idx3-ubyte");
            std :: vector<uint8_t> train_dataset_labels = read_dataset_labels("mnist/train-labels-idx1-ubyte");

            int epoch_iterations = 0;
            int input_batch_size = 0;

            std :: cout << "Epoch number of iterations\n";
            std :: cin >> epoch_iterations;

            std :: cout << "\nInput batch size\n";
            std :: cin >> input_batch_size;

            if(train_dataset.empty()) {
                throw std :: invalid_argument("Empty dataset");
            }

            if(train_dataset[0].size() != model.nodes_per_layer_list[0]) {
                throw std :: invalid_argument("Input layer nodes number doesn't correspond to data set element size");
            }

            double total_loss = 0.0;

            for(int i = 0; i < epoch_iterations; i++) {

                double partial_loss = 0.0;

                for(int j = 0; j < train_dataset.size(); j += input_batch_size) {
                    Matrix images_converted_to_batch_input = Matrix :: transpose(Matrix(train_dataset));

                    Matrix activated_output = model.forward(
                            images_converted_to_batch_input,
                            Matrix :: softmax_activation,
                            Matrix ::reLU_activation,
                            ML :: cross_entropy_loss_with_softmax_derived,
                            ML ::
                            );
                }
            }

            model.update_model_file();

            state = 0;
        }

        if(state == 4){

        }
    }
}
