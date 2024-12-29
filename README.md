Chronic Disease Prediction Using Machine Learning

Project Overview

This project focuses on using Machine Learning (ML) and Deep Learning (DL) techniques to predict chronic diseases such as diabetes, heart disease, lung cancer, and brain tumors. By leveraging datasets from various sources, the project aims to provide an efficient system for early detection and prevention of these diseases.

Features

Predicts multiple chronic diseases: Diabetes, Heart Disease, Lung Cancer, and Brain Tumor.

Uses advanced ML and DL models for high accuracy.

Easy-to-use web interface implemented with Streamlit.

Provides suggestions for treatment and prevention based on predictions.

System Requirements

Hardware

Processor: Intel Core i5 or higher

RAM: 8GB or higher

Storage: Minimum of 100GB HDD or 256GB SSD

Graphics Card: NVIDIA GeForce GTX 1050 or higher

Monitor: 1920x1080 resolution or higher

Software

Operating System: Windows 10 or Linux Ubuntu 18.04 or later

Python: Version 3.7 or higher

Python Libraries:

NumPy

Pandas

Matplotlib

Scikit-learn

TensorFlow

Keras

PyTorch

cv2

imutils

Web Framework: Streamlit

IDE: PyCharm or Jupyter Notebook

Installation

Step 1: Clone the Repository

$ git clone <repository-url>
$ cd <repository-folder>

Step 2: Install Dependencies

Create a virtual environment and install the required libraries:

$ python -m venv env
$ source env/bin/activate  # On Windows: env\Scripts\activate
$ pip install -r requirements.txt

Step 3: Run the Application
Run the deploy code.

Step 4: Input Data and Get Predictions

Use the web interface to input the required parameters for the disease you wish to predict.

View the results and suggested preventive measures.

Dataset Details:

Diabetes:

Source: Pima Indian Diabetes Dataset (Kaggle)

Features:

Pregnancies,Glucose,Insulin,BMI,Age

Heart Disease:

Source: Public Health Dataset (Kaggle)

Features:

Age, Sex, Chest Pain Type, Resting Blood Pressure, etc.

Lung Cancer:

Source: Histopathological Image Dataset (Kaggle)

Classes:

Benign lung tissue

Lung adenocarcinoma

Lung squamous cell carcinoma

Brain Tumor

Source: Br35H Brain Tumor Dataset (Kaggle)

Preprocessing:

Data Augmentation, Label Binarization

Models Used:

Random Forest:

-Diabetes: 91%

-Heart Disease: 98.53%

CNN (Sequential):

-Lung Cancer: 89%

CNN (VGG16):

-Brain Tumor: 89%

Future Work:

-Adding support for more chronic diseases.

-Improving accuracy with larger datasets and advanced models.

-Integrating cloud-based real-time predictions.

-Enhancing explainability of model predictions.# Chronic-Disease-Prediction
