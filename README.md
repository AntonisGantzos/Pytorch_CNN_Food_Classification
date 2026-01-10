# PyTorch Food Classification Project

A deep learning project that implements a Convolutional Neural Network (CNN) to classify images of food into three categories: **pizza**, **steak**, and **sushi**.

## Project Overview

This project demonstrates the end-to-end workflow of building an image classification model using PyTorch, from data loading to model evaluation and inference on custom images.

## Step-by-Step Logic

### 1. Environment Setup
- Import necessary libraries (PyTorch, torchvision, PIL, matplotlib, numpy)
- Configure device-agnostic code to utilize GPU if available, otherwise fall back to CPU

### 2. Data Acquisition
- Download a subset of the Food101 dataset containing only 3 classes (pizza, steak, sushi)
- This reduced dataset (10% of original) allows for faster experimentation
- Data is downloaded as a zip file and extracted to a local directory

### 3. Data Exploration
- Walk through the directory structure to understand data organization
- Visualize random sample images with their labels and dimensions
- Display images using PIL and matplotlib

### 4. Data Transformation
- Create transformation pipelines using `torchvision.transforms.Compose`
- **Basic transforms**: Resize images to 64x64, random horizontal flip, convert to tensor
- **Augmented transforms**: Include `TrivialAugmentWide` for automatic data augmentation
- Visualize original vs transformed images to verify transformations

### 5. Dataset and DataLoader Creation
- Use `torchvision.datasets.ImageFolder` to load images from directory structure
- Extract class names and class-to-index mappings
- Create `DataLoader` objects with batch size of 32 for efficient training
- Enable multi-worker data loading for performance

### 6. Model Architecture (TinyVGG)
- Implement a simplified VGG-style CNN architecture
- **Conv Block 1**: Two Conv2d layers with ReLU activation + MaxPool2d
- **Conv Block 2**: Two Conv2d layers with ReLU activation + MaxPool2d
- **Classifier**: Flatten layer followed by Linear layer for final predictions
- Use `torchinfo.summary()` to inspect model architecture

### 7. Training Infrastructure
- **train_step()**: Forward pass, loss calculation, backpropagation, optimizer step, accuracy tracking
- **test_step()**: Model evaluation without gradient computation
- **train()**: Main training loop that orchestrates training/testing over multiple epochs

### 8. Model Training - Experiment 1 (Without Augmentation)
- Train TinyVGG with simple transforms (resize + to tensor only)
- Use CrossEntropyLoss and Adam optimizer (lr=0.001)
- Train for 7 epochs and track metrics

### 9. Model Training - Experiment 2 (With Augmentation)
- Apply `TrivialAugmentWide` to training data for automatic augmentation
- Train a new TinyVGG instance with augmented data
- Compare training time and performance

### 10. Results Visualization
- Plot loss curves (train/test loss over epochs)
- Plot accuracy curves (train/test accuracy over epochs)
- Compare Model 0 (no augmentation) vs Model 1 (with augmentation) side by side

### 11. Custom Image Prediction
- Download a custom pizza image not in the dataset
- Preprocess the image:
  - Convert to float32 and scale to [0, 1]
  - Resize to 64x64
  - Add batch dimension
  - Move to appropriate device
- Run inference and convert logits to probabilities and labels
- Visualize prediction with the image

### 12. Prediction Utility Function
- Create `pred_and_plot_image()` helper function for easy inference on any image
- Handles all preprocessing, inference, and visualization in one call

## Key Concepts Demonstrated

| Concept | Description |
|---------|-------------|
| **Transfer Learning Prep** | Structure allows easy swap to pretrained models |
| **Data Augmentation** | Improves generalization by artificially increasing training data diversity |
| **Device Agnostic Code** | Seamlessly runs on CPU or GPU |
| **Modular Design** | Reusable training/testing functions |
| **Experiment Tracking** | Results dictionary for comparing models |

## Requirements

```
torch
torchvision
torchinfo
matplotlib
numpy
PIL
requests
tqdm
pandas
```

## Usage

1. Run all cells sequentially in the notebook
2. Models will be trained and evaluated automatically
3. Use `pred_and_plot_image()` to test on your own food images

## Future Improvements

- Increase model complexity or use pretrained models (ResNet, EfficientNet)
- Train on full Food101 dataset
- Implement learning rate scheduling
- Add model checkpointing and early stopping
- Use TensorBoard or Weights & Biases for experiment tracking
