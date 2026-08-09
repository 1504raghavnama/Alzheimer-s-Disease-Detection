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

