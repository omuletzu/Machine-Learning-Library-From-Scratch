# Machine Learning Library

## Overview
This library provides tools for configuring and training a neural network with customizable options. It supports loading pre-trained models, defining network architecture, and selecting activation functions and more.

## Features
- Configure the number of layers and perceptrons per layer.
- Load pre-trained model files for training or testing.
- Choose activation functions and other parameters.
- Includes a reader for MNIST database files.
- Supports gradient descent optimization for training.
- Provides matrix operations for computations.

## Usage
### Model Configuration
Users can:
- Select a model file containing weights and biases.
- Define the number of nodes and perceptrons per layer.
- Choose to train or test the model with a dataset.

### Training Process
1. Data is passed through the network with an activation function applied at each perceptron.
2. The output layer uses a specified activation function.
3. Gradient descent updates weights and biases after forward propagation.
4. This process is repeated for all dataset inputs for a configurable number of epochs.
5. The modified model file saves updated weights and biases.

## Configurable Options
- Activation function for hidden layers.
- Activation function for output layer.
- Cost function.
- Number of layers for a new model.
- Number of perceptrons per layer.
- Input training/testing file for the selected model.
- Learning rate and input batch matrix size for training

## Notes
After each training session, the original model file is updated with the corrected weights and biases.

<p align="center">
  <b>All options user can select</b><br>
  <img src="https://github.com/user-attachments/assets/68f2b7af-6667-4e58-861f-68c249d18ae9" width="350">
</p>

<p align="center">
  <b>Create New Model</b><br>
  <img src="https://github.com/user-attachments/assets/9051e2b2-0f18-456e-907b-ef6cb024fca8" width="300">
</p>

<p align="center">
  <b>Load Existing Model</b><br>
  <img src="https://github.com/user-attachments/assets/92ff29c4-efd3-47c2-a719-3d8f2ef26bcd" width="300">
</p>

<p align="center">
  <b>Training - Configurable Options</b><br>
  <img src="https://github.com/user-attachments/assets/71242b85-87d3-4a26-af70-b3e099255c2b" width="300">
</p>

<p align="center">
  <b>Training - Error Right After Start</b><br>
  <img src="https://github.com/user-attachments/assets/ef4a9e9b-a37c-4845-9460-fa2a91fee122" width="300">
</p>

<p align="center">
  <b>Training - Better Error Results At Random Time</b><br>
  <img src="https://github.com/user-attachments/assets/6f7ec4a7-a466-4b1f-a7e6-24ae5fcaea16" width="300">
</p>

<p align="center">
  <b>First Epoch - Final Error Result</b><br>
  <img src="https://github.com/user-attachments/assets/a429fb76-479d-4a17-b622-dbc1e98651c2" width="300">
</p>

<p align="center">
  <b>Last Epoch - Final Error Result</b><br>
  <img src="https://github.com/user-attachments/assets/09363070-a08a-483f-8fd4-d4bae971994a" width="300">
</p>



