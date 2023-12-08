# Celebrity Image Classification using CNNs

## Introduction

This script implements a machine learning model for classifying images of celebrities using convolutional neural networks (CNNs). The goal is to recognize and categorize images of five celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.

## Procedure

1. **Data Preparation**
   - The script assumes that the celebrity images are organized into directories for each celebrity within a common parent directory named `cropped`.
   - Images are loaded and resized to a common shape of (128, 128) pixels.

2. **Dataset Creation**
   - Separate lists are created for image filenames and corresponding labels for each celebrity.
   - A unified dataset (`X`) and label array (`y`) are created by iterating through each celebrity's images.

3. **Train-Test Split**
   - The dataset is split into training and testing sets using the `train_test_split` function from `sklearn`.
   - The split is performed with 80% of the data used for training and 20% for testing.

4. **Data Normalization**
   - The pixel values of images are normalized using `tf.keras.utils.normalize` to ensure consistent scaling across images.

5. **CNN Model Creation**
   - A sequential model is created using TensorFlow and Keras.
   - The model consists of convolutional layers, max-pooling layers, a flattening layer, and densely connected layers with dropout for regularization.
   - The output layer uses the softmax activation function for multi-class classification.

6. **Model Compilation**
   - The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function for training.
   - Accuracy is chosen as the metric to monitor during training.

7. **Model Training**
   - The model is trained on the training dataset using the `fit` method.
   - Training is performed for 30 epochs with a batch size of 128.
   - A portion (10%) of the training data is used as a validation set during training.

8. **Model Evaluation**
   - The trained model is evaluated on the test set to measure its accuracy and performance.

9. **Predictions on New Images**
   - The model is used to make predictions on a set of new images representing each celebrity.
   - The predictions are printed along with the filenames of the images.

## Key Findings

### Model Performance

- The accuracy of the model on the test set is reported.
- Classification report metrics (precision, recall, f1-score) are displayed, providing a detailed assessment of the model's performance for each class.

### Prediction on New Images

- The model's predictions on a set of new images are presented, showcasing its ability to generalize to unseen data.
- Predicted labels are printed alongside the filenames of the images.

### Model Architecture

- The architecture of the CNN model is summarized, providing insights into the layers and structure used for image classification.

