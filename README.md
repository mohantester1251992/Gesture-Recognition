# Gesture Recognition Project

This project aims to build a **3D Convolutional Neural Network (Conv3D)** capable of recognizing five different hand gestures. The gestures are identified from video sequences, and the model will be trained to predict the correct gesture based on this data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The task is to recognize five different hand gestures using deep learning, specifically with the use of a Conv3D model. The dataset consists of video sequences of gestures, and the goal is to develop a model that correctly classifies each gesture. The project involves several steps:
1. Preprocessing video data.
2. Building a Conv3D model.
3. Training the model with the processed video data.
4. Evaluating the model's performance.

## Dataset
The dataset contains video sequences of individuals performing five distinct gestures. Each gesture corresponds to one class label. The dataset should be organized into separate folders for each gesture, with videos saved in a format that can be processed using Python's libraries.

**Data format:**
- Each gesture is represented by a folder, and each folder contains multiple videos representing that gesture.
- The videos need to be preprocessed into frames for model input.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/isidharthrai/Gesture-Recognition-Project-Upgrad.git
   cd Gesture-Recognition-Project-Upgrad
   ```

2. **Install dependencies:**
   Make sure you have Python installed. Then, install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preprocessing:**
   The video data should be preprocessed by extracting frames from the videos. Each video is converted into a sequence of frames, resized, and normalized before being passed into the model.

2. **Model Training:**
   To train the model, open the Jupyter notebook `Gesture_Recognition_assignment.ipynb` and execute the cells in sequence. The notebook includes detailed steps for:
   - Loading and preprocessing the data.
   - Defining the Conv3D architecture.
   - Training and evaluating the model.

3. **Model Evaluation:**
   After training, the model will be evaluated using a test dataset. Evaluation metrics include accuracy and confusion matrix.

4. **Making Predictions:**
   Once the model is trained, you can use it to make predictions on new video sequences.

## Model Architecture
The Conv3D model is designed to capture spatiotemporal features from video sequences. Here are the key components:
- **Conv3D layers** to extract features from the video frames.
- **MaxPooling3D layers** to reduce dimensionality.
- **Fully connected layers** for classification.
- **Softmax output** for predicting the gesture class.

## Training the Model
Training is performed using the preprocessed video frames. The following steps are involved in training:
1. Loading video frames and preparing the training set.
2. Compiling the model with appropriate loss functions and optimizers.
3. Fitting the model with the training data and monitoring performance using validation data.

The model can be fine-tuned using hyperparameter optimization, batch normalization, and dropout layers to improve performance.

## Evaluation
The model is evaluated on the test set using metrics such as:
- **Accuracy**: Overall correctness of predictions.
- **Confusion Matrix**: Breakdown of predicted vs actual classes.
- **Precision, Recall, and F1-Score**: For a more detailed performance assessment.

## Results
- **Training Accuracy**: 0.99
- **Validation Accuracy**: 0.94
- The trained model successfully classifies five different gestures with reasonable accuracy.

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes. Ensure that your changes align with the project goals and include necessary documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Code by [mohantester1251992@gmail.com]
