# Handwritten Character Recognition (OCR) using CNN

A deep learning-based Optical Character Recognition (OCR) system that recognizes and reads handwritten characters and words from images using Convolutional Neural Networks.

## Overview

This project implements a CNN-based character recognition system capable of identifying 35 different classes (A-Z and 0-9) from handwritten text images. The system processes images, segments individual characters using contour detection, and predicts each character to form complete words.

## Features

- **Multi-character Recognition**: Recognizes uppercase letters (A-Z) and digits (0-9)
- **Image Preprocessing**: Automatic image enhancement using OpenCV
- **Character Segmentation**: Intelligent contour-based character extraction
- **Deep Learning Model**: Custom CNN architecture optimized for character recognition
- **Word Formation**: Combines individual character predictions to form complete words
- **Model Checkpointing**: Saves best performing model during training
- **Early Stopping**: Prevents overfitting with patience-based training termination

## Technology Used

### Deep Learning & Machine Learning
- **TensorFlow** - Deep learning framework for building and training neural networks
- **Keras** - High-level neural networks API (integrated with TensorFlow)
- **scikit-learn** - Machine learning library for preprocessing and encoding

### Computer Vision
- **OpenCV (cv2)** - Image processing and computer vision operations
- **imutils** - Convenience functions for OpenCV operations

### Data Processing & Analysis
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis

### Visualization
- **Matplotlib** - Plotting and data visualization
- **Seaborn** - Statistical data visualization

### Development Environment
- **Jupyter Notebook** - Interactive development environment
- **Google Colab** - Cloud-based notebook with GPU support (T4)

### Model Components
- **Convolutional Neural Networks (CNN)** - Core architecture for image recognition
- **AdamW Optimizer** - Adaptive learning rate optimization with weight decay
- **Dropout Layers** - Regularization technique to prevent overfitting
- **MaxPooling** - Spatial downsampling for feature extraction
- **Categorical Crossentropy** - Loss function for multi-class classification

### Data Handling
- **Kaggle API** - Dataset download and management
- **Label Binarizer** - One-hot encoding for categorical labels
- **tqdm** - Progress bar for loops and iterations

## Dataset

The project uses the [Handwritten Characters Dataset](https://www.kaggle.com/datasets/vaibhao/handwritten-characters) from Kaggle, which contains:
- Training images organized by character class
- Validation images for model evaluation
- Test images for final predictions
- 35 character classes (excluding special characters: #, $, &, @)

## Project Structure

```
├── data/
│   ├── Train/          # Training images organized by character
│   ├── Test/           # Test images for predictions
│   └── Val/            # Validation images
├── best_model.keras    # Saved best model weights
├── Optical.ipynb       # Main Jupyter notebook
└── README.md           # Project documentation
```

## Model Architecture

The CNN model consists of:

```
Input Layer (32x32x1 grayscale images)
    ↓
Conv2D (32 filters, 3x3) + ReLU + MaxPooling (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU + MaxPooling (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU + MaxPooling (2x2)
    ↓
Dropout (0.25)
    ↓
Flatten
    ↓
Dense (128 units) + ReLU
    ↓
Dropout (0.2)
    ↓
Dense (35 units) + Softmax
```

### Model Specifications

- **Input Shape**: 32x32x1 (grayscale images)
- **Output Classes**: 35 (A-Z and 0-9)
- **Total Parameters**: ~500K trainable parameters
- **Optimizer**: AdamW (learning_rate=1e-3, weight_decay=1e-4)
- **Loss Function**: Categorical Crossentropy



## Key Functions

### `get_letters(image_path)`
Processes an image and extracts individual characters:
- Converts to grayscale
- Applies binary thresholding
- Detects and sorts contours
- Extracts character regions
- Returns predictions and processed image

### `sort_contours(cnts, method)`
Sorts detected contours for proper character ordering:
- Supports left-to-right and top-to-bottom sorting
- Uses bounding box coordinates
- Returns sorted contours and bounding boxes

### `get_word(letter)`
Combines individual character predictions into a complete word.

## Training Configuration

- **Epochs**: 30 (with early stopping)
- **Batch Size**: 32
- **Image Size**: 32x32 pixels
- **Training Samples**: ~2000 per class (limited for balance)
- **Validation Split**: Separate validation set
- **Callbacks**:
  - EarlyStopping (patience=10, monitor='val_loss')
  - ModelCheckpoint (monitor='val_accuracy', save_best_only=True)

## Results

The model achieves high accuracy on the validation set with:
- Training and validation accuracy(92 %) curves showing good convergence
- Loss curves indicating proper learning without overfitting
- Successful word recognition on test images

## Image Preprocessing Pipeline

1. **Grayscale Conversion**: Converts RGB images to single channel
2. **Binary Thresholding**: Separates characters from background
3. **Dilation**: Enhances character boundaries
4. **Contour Detection**: Identifies individual characters
5. **Bounding Box Extraction**: Isolates character regions
6. **Resizing**: Normalizes to 32x32 pixels
7. **Normalization**: Scales pixel values to [0, 1]

## Limitations

- Excluded special characters (#, $, &, @) from recognition
- Requires clear character separation for accurate segmentation
- Limited to uppercase letters and digits
- Performance depends on handwriting clarity and image quality

