# 🧠 Alzheimer’s Disease Detection Using Deep Learning

## 📌 Project Overview
This project focuses on the automated detection of Alzheimer’s Disease from brain MRI scans using deep learning techniques. The system classifies MRI images into clinically relevant stages, enabling early diagnosis support and scalable medical decision assistance.

The model leverages Convolutional Neural Networks (CNNs) to extract spatial features from MRI data and learn disease-specific patterns with high reliability.

---

## 🧠 Problem Statement
Alzheimer’s Disease is a progressive neurodegenerative disorder where early detection is critical but challenging due to subtle anatomical changes in brain structure. Traditional diagnosis methods are time-consuming and subjective. This project aims to develop an automated, data-driven solution to assist clinicians with accurate and early-stage detection.

---

## 🚀 Key Features
- Multi-class classification of Alzheimer’s stages from MRI scans  
- Automated feature extraction using CNN architecture  
- Robust preprocessing pipeline for medical imaging data  
- Scalable and reproducible training workflow  

---

## 🗂 Dataset Information
- **Dataset Type:** Brain MRI Images  
- **Classes:**  
  - Non-Demented  
  - Very Mild Demented  
  - Mild Demented  
  - Moderate Demented  

- **Total Images:** ~6,400+ MRI scans  
- **Image Format:** JPEG / PNG  
- **Data Split:**  
  - Training: 70%  
  - Validation: 15%  
  - Testing: 15%  

---

## 🧠 Model Architecture
- **Base Architecture:** Convolutional Neural Network (CNN)  
- **Layers Used:**  
  - Convolution + ReLU  
  - Max Pooling  
  - Dropout (to reduce overfitting)  
  - Fully Connected Dense Layers  

- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam  

---

## 📊 Performance Snapshot
- **Training Accuracy:** ~96–98%  
- **Validation Accuracy:** ~93–95%  
- **Inference Time:** < 50 ms per image  
- **Overfitting Control:** Dropout + Data Augmentation  


---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **Frameworks & Libraries:**  
  - TensorFlow / Keras  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - OpenCV

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/alzheimer-detection.git
cd alzheimer-detection
pip install -r requirements.txt
```
---

## 📈 Results Summary
The deep learning model successfully learns discriminative spatial patterns from brain MRI scans corresponding to different stages of Alzheimer’s Disease. The training process shows stable convergence with high classification confidence across classes, indicating strong feature representation and robustness. Overall, the results validate the effectiveness of CNN-based approaches for automated Alzheimer’s disease detection and medical image analysis.
