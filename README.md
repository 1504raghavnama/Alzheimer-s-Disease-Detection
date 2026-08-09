# 🧠 Alzheimer's Disease Detection Using Deep Learning

<p align="center">
  <img src="assets/alzheimers-banner.png" alt="Alzheimer's Disease Detection" width="850"/>
</p>

<p align="center">
  <b>Multi-Class Brain MRI Classification using CNNs and ResNet50 Transfer Learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Metrics-F7931E?logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/Computer%20Vision-MRI-purple"/>
  <img src="https://img.shields.io/badge/Medical%20AI-Research-green"/>
</p>

---

## 📌 Project Overview

This project develops a deep learning-based system for **multi-class classification of Alzheimer's-related dementia stages from brain MRI images**.

The project implements and compares two approaches:

1. **Baseline Convolutional Neural Network (CNN)** built from scratch.
2. **ResNet50 Transfer Learning** using ImageNet-pretrained weights with partial fine-tuning and a custom classification head.

The complete workflow covers dataset organization, stratified data splitting, image preprocessing, augmentation, generator-based training, transfer learning, model optimization, and comprehensive evaluation using class-wise and multi-class performance metrics.

> **Note:** This project is intended as a machine learning research/educational system and should not be considered a clinically validated diagnostic tool.

---

## 🎯 Problem Statement

Alzheimer's disease is a progressive neurodegenerative condition associated with structural changes in the brain. Distinguishing different stages from MRI scans can be challenging because the visual differences between certain classes can be subtle.

The objective of this project is to investigate whether deep learning models can automatically learn discriminative spatial patterns from brain MRI images and classify them into four Alzheimer's-related categories.

### Classification Classes

| Class | Description |
|---|---|
| 🟢 **NonDemented** | MRI scans without dementia classification |
| 🟡 **VeryMildDemented** | Very mild dementia stage |
| 🟠 **MildDemented** | Mild dementia stage |
| 🔴 **ModerateDemented** | Moderate dementia stage |

---

# 🗂️ Dataset

The dataset consists of **33,984 brain MRI images** distributed across four classes.

### Class Distribution

| Class | Images |
|---|---:|
| NonDemented | 9,600 |
| MildDemented | 8,960 |
| VeryMildDemented | 8,960 |
| ModerateDemented | 6,464 |
| **Total** | **33,984** |

The images are organized into class-specific directories and converted into a structured Pandas DataFrame containing image file paths and corresponding class labels.
---

## 📊 Dataset Visualization

The visualization step verifies that MRI images are loaded correctly and that the corresponding class labels are mapped properly before model training.

<p align="center">
  <img src="assets/sample_mri_images.png" alt="Sample Brain MRI Images" width="900"/>
</p>

The sample grid shows MRI scans from all four classes, providing a visual verification of the dataset labeling and image-loading pipeline.
---

# 🔬 Machine Learning Pipeline

The project follows an end-to-end medical image classification pipeline, starting from raw MRI images and progressing through dataset preparation, stratified partitioning, image preprocessing, model training, and comprehensive performance evaluation.

```text
                 Brain MRI Dataset
                        │
                        ▼
              Dataset Organization
                        │
                        ▼
              Pandas DataFrame
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
      Image File Paths          Labels
             │                     │
             └──────────┬──────────┘
                        ▼
              Stratified Splitting
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
     Training       Validation         Test
     24,553           4,333           5,098
        │               │               │
        ▼               ▼               ▼
 Augmentation        Rescaling       Rescaling
        │               │               │
        └───────────────┴───────────────┘
                        ▼
                ImageDataGenerator
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Baseline CNN          ResNet50 Transfer
                             Learning + Fine-Tuning
             │                     │
             └──────────┬──────────┘
                        ▼
                 Model Prediction
                        │
                        ▼
              Comprehensive Evaluation
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
    Classification   Confusion        ROC-AUC
      Metrics         Matrix           Curves
```
---


# ⚙️ Data Preparation

The MRI dataset was transformed into a structured **Pandas DataFrame** containing image file paths and their corresponding class labels. This tabular representation was then used to construct the training, validation, and test pipelines.

### 📊 Dataset Partition

A **stratified two-stage split** was implemented using `train_test_split()` with a fixed random seed to preserve class proportions across the datasets.

| Dataset | Images | Purpose |
|---|---:|---|
| **Training** | 24,553 | Model parameter learning |
| **Validation** | 4,333 | Hyperparameter tuning and generalization monitoring |
| **Testing** | 5,098 | Final evaluation on unseen data |
| **Total** | **33,984** | |

### 🔄 Preprocessing Strategy

- **Image Resizing:** All MRI images are resized to `128 × 128` pixels.
- **Normalization:** Pixel intensities are rescaled from `0–255` to `0–1`.
- **Training Augmentation:** Random rotation, zoom, and horizontal flipping are applied to increase training-data diversity.
- **Validation/Test Processing:** Only resizing and normalization are applied to prevent artificial modifications during evaluation.
- **Batch Processing:** Images are loaded through TensorFlow/Keras `ImageDataGenerator` with a batch size of `32`.

The test set remains completely isolated from model training and is used only after training is completed for unbiased performance evaluation.
---

# 🧪 Image Preprocessing & Augmentation

Before being passed to the neural networks, MRI images undergo a standardized preprocessing pipeline to ensure consistent input dimensions and normalized pixel values.

### 🖼️ Image Preprocessing

| Operation | Configuration | Purpose |
|---|---|---|
| **Resizing** | `128 × 128` | Standardizes image dimensions |
| **Rescaling** | `1./255` | Converts pixel values from `0–255` to `0–1` |
| **Batching** | `32 images/batch` | Enables memory-efficient mini-batch training |

### 🔄 Training Data Augmentation

The training generator applies controlled transformations to increase input diversity and improve model generalization:

- **Rotation:** Up to `20°`
- **Zoom:** Up to `20%`
- **Horizontal Flip:** Random horizontal transformations
- **Normalization:** Pixel-value rescaling to `[0, 1]`

These transformations generate varied versions of training samples without modifying the original dataset.

### 🧪 Validation & Test Processing

Validation and test images undergo **only resizing and normalization**. Augmentation is intentionally excluded from these datasets so that model performance is measured on data that has not been artificially transformed.

The preprocessing pipeline is implemented using TensorFlow/Keras `ImageDataGenerator` and `flow_from_dataframe()`.
---

# 🧠 Baseline CNN

A custom **Convolutional Neural Network (CNN)** was developed from scratch to establish a baseline for the Alzheimer's MRI classification task. The architecture performs hierarchical feature extraction through convolution and pooling layers, followed by fully connected layers for four-class classification.

### 🏗️ Architecture

```text
Input MRI
128 × 128 × 3
      │
      ▼
Conv2D
32 Filters • 3 × 3 Kernel • ReLU
      │
      ▼
MaxPooling2D
2 × 2 Pool
      │
      ▼
Conv2D
64 Filters • 3 × 3 Kernel • ReLU
      │
      ▼
MaxPooling2D
2 × 2 Pool
      │
      ▼
Flatten
      │
      ▼
Dense
128 Neurons • ReLU
      │
      ▼
Dense
4 Neurons • Softmax
      │
      ▼
Alzheimer's Class
```
---

# 🚀 ResNet50 Transfer Learning

To improve upon the baseline CNN, the project uses **ResNet50 with ImageNet-pretrained weights** as the feature extraction backbone. Transfer learning allows the model to reuse visual representations learned from a large-scale image dataset and adapt them to the Alzheimer's MRI classification task.

### 🏗️ Architecture

```text
Input MRI
128 × 128 × 3
      │
      ▼
ImageNet-Pretrained ResNet50
      │
      ▼
Partial Fine-Tuning
Last 10 Layers Trainable
      │
      ▼
Flatten
      │
      ▼
Dense Layer
256 Neurons • ReLU
      │
      ▼
Dropout
0.5
      │
      ▼
Dense Layer
4 Neurons • Softmax
      │
      ▼
Alzheimer's Class
```
---
# ⚙️ Training Configuration

The models were trained using TensorFlow/Keras with a configuration designed for stable optimization and controlled generalization.

| Configuration | Setting |
|---|---|
| **Framework** | TensorFlow / Keras |
| **Input Shape** | `128 × 128 × 3` |
| **Batch Size** | `32` |
| **Maximum Epochs** | `20` |
| **Optimizer** | Adam |
| **Learning Rate** | `0.0001` |
| **Loss Function** | Categorical Cross-Entropy |
| **Output Activation** | Softmax |
| **Number of Classes** | `4` |

### 🛡️ Training Controls

#### EarlyStopping

`EarlyStopping` monitors validation performance during training and prevents unnecessary additional epochs when the model stops improving. The callback also restores the best-performing model weights.

#### ReduceLROnPlateau

`ReduceLROnPlateau` dynamically decreases the learning rate when validation performance reaches a plateau. This enables smaller optimization steps during later stages of training and helps the model continue refining its parameters.

### 🔄 Model Training Strategy

The **baseline CNN** is trained from scratch to establish a reference point, while the **ResNet50 model** starts from ImageNet-pretrained representations and performs partial fine-tuning on the MRI dataset.

The same validation pipeline is used to monitor generalization throughout training, while the held-out test set remains untouched until the final evaluation stage.

---

# 🧪 Model Evaluation

After training, the final ResNet50 model is evaluated exclusively on the **5,098-image held-out test set**. The evaluation pipeline generates class predictions and probability scores, which are then used to assess both overall performance and class-specific behavior.

### 📊 Evaluation Metrics

The model is evaluated using multiple complementary metrics:

| Metric | Purpose |
|---|---|
| **Accuracy** | Measures the overall proportion of correctly classified MRI images |
| **Precision** | Measures the correctness of positive predictions for each class |
| **Recall** | Measures how effectively samples from each class are identified |
| **F1-score** | Provides a balance between precision and recall |
| **Confusion Matrix** | Analyzes class-wise correct and incorrect predictions |
| **ROC-AUC** | Measures the model's ability to discriminate between classes across thresholds |

### 🔲 Confusion Matrix — Prediction Counts

The raw confusion matrix shows the number of test samples assigned to each predicted class compared with their actual class.

<p align="center">
  <img src="assets/confusion_matrix_counts.png" alt="Confusion Matrix Counts" width="700"/>
</p>

### 🔲 Normalized Confusion Matrix

The normalized confusion matrix represents class-wise prediction proportions, making it easier to compare classification behavior across classes with different sample distributions.

<p align="center">
  <img src="assets/confusion_matrix_normalized.png" alt="Normalized Confusion Matrix" width="700"/>
</p>

### 📈 Multi-Class ROC Analysis

A **One-vs-Rest (OvR)** strategy is used to generate ROC curves for the four Alzheimer's-related classes. The ROC-AUC metric evaluates how effectively the model separates each class from the remaining classes across different classification thresholds.

<p align="center">
  <img src="assets/roc_curves.png" alt="Multi-Class ROC Curves" width="850"/>
</p>

This multi-level evaluation provides a more complete view of model behavior than relying on accuracy alone, particularly for identifying class-specific errors and differences in discriminative performance.
---

# 🏆 Results & Performance

The final ResNet50 transfer-learning model was evaluated on **5,098 unseen MRI images** from the held-out test set.

### 📊 Overall Performance

| Metric | Score |
|---|---:|
| **Test Accuracy** | **74.07%** |
| **Macro Precision** | **76.05%** |
| **Macro Recall** | **75.88%** |
| **Macro F1-score** | **75.93%** |
| **Macro ROC-AUC** | **0.9273** |

### 📋 Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| **MildDemented** | 0.80 | 0.74 | 0.77 |
| **ModerateDemented** | **0.98** | **1.00** | **0.99** |
| **NonDemented** | 0.70 | 0.72 | 0.71 |
| **VeryMildDemented** | 0.56 | 0.58 | 0.57 |

### 📈 Class-wise ROC-AUC

| Class | ROC-AUC |
|---|---:|
| **MildDemented** | **0.95** |
| **ModerateDemented** | **1.00** |
| **NonDemented** | **0.91** |
| **VeryMildDemented** | **0.85** |

### 🔍 Performance Insights

The model demonstrates strong class-separation capability with a **0.9273 macro ROC-AUC**. The **ModerateDemented** class achieves the strongest classification performance, while **VeryMildDemented** presents the greatest challenge, with comparatively lower precision, recall, F1-score, and ROC-AUC.

The difference in class-wise performance highlights the difficulty of distinguishing subtle patterns associated with the very mild stage and motivates further investigation into class-aware training strategies and more advanced architectures.
---
# 🛠️ Tech Stack

### Programming & Deep Learning

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Models** | CNN, ResNet50, Transfer Learning |
| **Data Processing** | NumPy, Pandas |
| **Computer Vision** | OpenCV |
| **Machine Learning / Evaluation** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Development Environment** | Jupyter Notebook |

### Core Techniques

```text
Computer Vision
      │
      ├── Image Preprocessing
      ├── Data Augmentation
      ├── CNN Feature Extraction
      ├── Transfer Learning
      ├── Partial Fine-Tuning
      ├── Dropout Regularization
      ├── Learning-Rate Scheduling
      └── Multi-Class Classification
```
