#include <iostream>
#include <string>
#include "ML.h"
#include "Matrix.h"

int main() {

    int state = 0;

    ML model = ML();

    while(true){
        if(state == 0){
            std :: cout << "Choose an option\n";
            std :: cout << "\t 1. Create new model file\n";
            std :: cout << "\t 2. Load model file\n";
            std :: cout << "\t 3. Train model\n";
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

            std :: ofstream fout = std :: ofstream(filename);

            if(model.load_model_file(filename)){
                std :: cout << "Loaded model - " + filename + " -\n";
            }
            else{
                std :: cout << "Failed loading - " + filename + " -\n";
            }
        }

        if(state == 3){
            
        }
    }
}
