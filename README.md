# ResNet50 Image Classification and Analysis

This project involves training a ResNet50 model for image classification, analyzing feature maps, and performing Principal Component Analysis (PCA) on feature representations. It is structured around loading and preprocessing image datasets, training the model, evaluating its performance, and conducting further analysis on the feature maps and PCA results.

## Features

### Dataset Management
- Images are loaded and split into training, validation, and test datasets.
- Data is augmented using transformations to improve model robustness.

### Model Training
- A ResNet50 model is trained on the preprocessed dataset.
- The model's performance is tracked over multiple epochs.

### Evaluation and Softmax Comparison
- The trained model is evaluated on the test dataset.
- Predictions are compared with previously saved softmax outputs for consistency.

### Feature Map Analysis
- Feature map statistics are computed to understand the internal representations of the model.

### Principal Component Analysis
- PCA is applied to the model's feature representations for dimensionality reduction and visualization.
