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

        if(state == 3 || state == 4){
            std :: vector<std :: vector<double>> dataset;
            std :: vector<uint8_t> dataset_labels;
            bool model_training = false;

            if(state == 3){
                dataset = read_dataset("mnist/train-images-idx3-ubyte");
                dataset_labels = read_dataset_labels("mnist/train-labels-idx1-ubyte");
                model_training = true;
            }
            else{
                dataset = read_dataset("mnist/t10k-images-idx3-ubyte");
                dataset_labels = read_dataset_labels("mnist/t10k-labels-idx1-ubyte");
            }

            int epoch_iterations = 0;
            int input_batch_size = 0;
            double learning_rate = 0.0;

            std :: cout << "Epoch number of iterations:\n";
            std :: cin >> epoch_iterations;

            std :: cout << "\nInput batch size:\n";
            std :: cin >> input_batch_size;

            std :: cout << "\nLearning rate:\n";
            std :: cin >> learning_rate;

            if(dataset.empty()) {
                throw std :: invalid_argument("Empty dataset");
            }

            if(dataset[0].size() != model.nodes_per_layer_list[0]) {
                throw std :: invalid_argument("Input layer nodes number doesn't correspond to data set element size");
            }

            std :: ofstream loss_log("loss_log.txt");

            double total_loss = 0.0;

            for(int i = 0; i < epoch_iterations; i++) {

                double partial_loss = 0.0;

                for(int j = 0; j < dataset.size(); j += input_batch_size) {
                    Matrix images_converted_to_batch_input = Matrix :: transpose(
                            Matrix(std :: vector<std :: vector<double>>(dataset.begin() + j, dataset.begin() + j + input_batch_size))
                            );

                    Matrix expected_outputs = Matrix(model.nodes_per_layer_list[model.nodes_per_layer_list.size() - 1], input_batch_size);

                    for(int k = 0; k < input_batch_size; k++){
                        int label = dataset_labels[j + k];
                        expected_outputs.set(label, k, 1.0);
                    }

                    Matrix activated_output = model.forward(
                            images_converted_to_batch_input,
                            Matrix :: softmax_activation,   //final activation
                            Matrix ::reLU_activation,       //hidden activation
                            Matrix :: softmax_derivative,   //final derivative
                            Matrix :: reLU_derivative,      //hidden derivative
                            ML :: cross_entropy_loss_function_with_softmax, //loss cost
                            ML :: cross_entropy_loss_with_softmax_derived,  //lost cost derivative
                            expected_outputs.get_matrix(),
                            learning_rate,
                            model_training
                            );

                    for(int k = 0; k < input_batch_size; k++){
                        partial_loss += activated_output.at(0, k);
                    }
                }

                partial_loss /= dataset.size();
                total_loss += partial_loss;

                loss_log << "For iteration " << i << " loss " << partial_loss << "\n";
            }

            model.update_model_file();

            total_loss /= epoch_iterations;

            loss_log << "Final loss " << total_loss << "\n";

            state = 0;
        }
    }
}
