# Emotion Detection System

## Overview

The Emotion Detection System is a Deep Learning-based application that detects human emotions from facial images. The project uses a Convolutional Neural Network (CNN) trained on the RAF-DB (Real-world Affective Faces Database) dataset to classify facial expressions into different emotion categories.

The application provides a graphical user interface (GUI) that allows users to upload images and predict emotions in real time.

---

## Features

* Facial emotion recognition using CNN
* Face detection using Haar Cascade Classifier
* GUI-based image upload and prediction
* Supports multiple emotion categories
* Pre-trained model for quick inference
* Built using TensorFlow, OpenCV, and Tkinter

---

## Emotion Classes

The model can classify the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Tkinter
* Scikit-Learn

---

## Dataset

This project utilizes the RAF-DB (Real-world Affective Faces Database) dataset for training and evaluation.

Dataset Components:

* Facial Images
* Training Labels
* Testing Labels

Dataset Structure:

```text
archive/
├── DATASET/
├── train_labels.csv
└── test_labels.csv
```

---

## CNN Architecture

The Emotion Detection model is based on a Convolutional Neural Network (CNN) consisting of:

1. Convolution Layers
2. ReLU Activation Functions
3. Max Pooling Layers
4. Dropout Layers
5. Fully Connected Dense Layers
6. Softmax Output Layer

Model Workflow:

```text
Input Image
      ↓
Face Detection
      ↓
Convolution Layers
      ↓
Activation (ReLU)
      ↓
Max Pooling
      ↓
Flatten
      ↓
Dense Layers
      ↓
Emotion Prediction
```

---

## Project Structure

```text
Emotion_Detection-Main/
└── Emotion_detection_main/
    ├── gui.py
    ├── model_creation.ipynb
    ├── model_a1.json
    ├── model_weights1.h5
    ├── haarcascade_frontalface_default.xml
    ├── README.md
    ├── requirements.txt
    ├── venv/
    └── archive/
        ├── DATASET/
        ├── train_labels.csv
        └── test_labels.csv
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For Git Bash:

```bash
source venv/Scripts/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install tensorflow opencv-python numpy pandas matplotlib pillow scikit-learn
```

---

## Running the Application

Run the GUI application:

```bash
python gui.py
```

Steps:

1. Launch the application.
2. Upload an image.
3. Click the Detect button.
4. View the predicted emotion.

---

## Training Information

Training Parameters:

* Image Size: 48 × 48
* Batch Size: 64
* Epochs: 15

Training Process:

1. Load RAF-DB dataset.
2. Preprocess images.
3. Generate batches using ImageDataGenerator.
4. Train CNN model.
5. Validate model performance.
6. Save model architecture and weights.

---

## Current Status

### Completed

* Project Environment Setup
* Dataset Integration (RAF-DB)
* CNN Architecture Study
* Model Training Analysis
* Emotion Detection GUI

### In Progress

* Attendance System Integration
* Face Recognition Module
* CSV Attendance Logging

### Planned Enhancements

* Student Attendance System
* Emotion-Based Attendance Analytics
* Real-Time Webcam Detection
* Attendance Reports in CSV/Excel Format
* Time-Based Attendance Restrictions

---

## Future Scope

* Multi-face emotion detection
* Real-time video emotion analysis
* Student attendance automation
* Emotion statistics dashboard
* Cloud deployment

---

## Internship Information

This project is being developed as part of the Elevance Skills AI & Machine Learning Internship Program.

---

## Author

**Minakshi Kaushik**

Data Science Intern
