import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
import cv2
import os

def diabetes_prediction():
    diabetes_bg_img = """ 
    <style>
        [data-testid="stAppViewContainer"]{
            background-image: url("https://www.westchesterhealth.com/blog/wha-media/uploads/2019/09/diabetes2.jpg");
            background-size: cover;
        }
        footer {
	    visibility: hidden;
	    }
        #MainMenu{
            visibility: hidden;
        }
        [data-testid="stHeader"] {
            display: none;
        } 
    </style>
    """
    st.markdown(diabetes_bg_img,unsafe_allow_html=True)

    # Load the diabetes dataset
    data = pd.read_csv(r"C:\Users\hp\Desktop\Major_project\ChronicDiseasePrediction\diabetes_disease\populated_dataset.csv")
    # splitting columns
    y=data['Outcome']
    data=data.drop(['Outcome'],axis=1)
    X=data
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                  random_state=0,
                                                  stratify = y)
    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)*100

    def get_user_input():
        with st.form(key="diabetes_form"):
            pregnancies = st.number_input('No. of preg', 0,17, 2)
            glucose = st.number_input('What is your plasma glucose concentration?', 0,200, 140)
            insulin = st.number_input('Enter your insulin 2-Hour serum in mu U/ml', 0,846, 120 )
            BMI = st.number_input('What is your Body Mass Index?', 0,67, 15)
            age = st.number_input('Enter your age', 21,88, 45 )
            user_data = {
                    'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'Insulin': insulin,
                    'BMI': BMI,
                    'Age': age,
                    }
            features = pd.DataFrame(user_data, index=[0])
            submit = st.form_submit_button("Get Result")
            if submit:
                return features

    user_input = get_user_input()
    if user_input is not None:
            st.write('Accuracy: {0:.2f} %'.format(accuracy))
            prediction = clf.predict(user_input)
            if prediction == 1:
                st.error("You either have diabetes or are likely to have it. Please visit the doctor as soon as possible.")
            else:
                st.success('Hurray! You are diabetes FREE.')

def heart_disease_prediction():
    heart_bg_img = """ 
    <style>
        [data-testid="stAppViewContainer"]{
            background-image: url("https://d2jx2rerrg6sh3.cloudfront.net/image-handler/picture/2021/10/shutterstock_1682046427-1.jpg");
            background-size: cover;
        }
        footer {
	    visibility: hidden;
	    }
        #MainMenu{
            visibility: hidden;
        }
        [data-testid="stHeader"] {
            display: none;
        } 
    </style>
    """
    st.markdown(heart_bg_img,unsafe_allow_html=True)

    # Load the heart dataset
    df = pd.read_csv(r"C:\Users\hp\Desktop\Major_project\ChronicDiseasePrediction\heart_disease\preprocessed_dataset.csv")
    # splitting columns
    X = df.drop(columns='condition')
    y = df['condition']
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)*100

    def get_user_input():
        with st.form(key="heart_disease_form"):
            age = st.number_input('Enter your age',0)
            sex = st.number_input('Enter your gender(1-male,0-female)',0,1)
            cp = st.number_input('Enter if you are having chestpain or not(0-1)', 0,1)
            trestbps = st.number_input('Enter your blood pressure',0)
            restecg = st.number_input('Enter your resting electrocardiographic results',0)
            thalach = st.number_input('Enter your maximum heart rate achieved',0)
            exang = st.number_input('Enter if exercise induced angina(0-1)',0,1)
            oldpeak = st.number_input('Enter ST depression induced by exercise relative to rest',0)
            slope = st.number_input('Enter the slope of the peak exercise ST segment(0-2)', 0,2)
            ca = st.number_input('Enter number of major vessels (0-3) colored by flourosopy', 0,3)
            thal = st.number_input('Enter thalium stress result(0-7)',0,7)
            user_data = {
                    'age': age,
                    'sex': sex,
                    'cp': cp,
                    'trestbps': trestbps,
                    'restecg': restecg,
                    'thalach': thalach,
                    'exang': exang,
                    'oldpeak': oldpeak,
                    'slope' : slope,
                    'ca' : ca,
                    'thal' : thal
                    }
            features = pd.DataFrame(user_data, index=[0])
            submit = st.form_submit_button("Get Result")
            if submit:
                return features

    user_input = get_user_input()
    if user_input is not None:
            st.write('Accuracy: {0:.2f} %'.format(accuracy))
            prediction = clf.predict(user_input)
            if prediction == 1:
                st.error("You either have heart problem or are likely to have it. Please visit the doctor as soon as possible.")
            else:
                st.success("Hurray! You don't have any heart problem.")

def lung_cancer_prediction():
    lung_cancer_bg_img = """ 
    <style>
        [data-testid="stAppViewContainer"]{
            background-image: url("https://img.freepik.com/free-photo/stethoscope-with-lung-shape-desk_23-2148533070.jpg");
            background-size: cover;
        }
        footer {
	    visibility: hidden;
	    }
        #MainMenu{
            visibility: hidden;
        }
        [data-testid="stHeader"] {
            display: none;
        } 
    </style>
    """
    st.markdown(lung_cancer_bg_img,unsafe_allow_html=True)
    img = st.file_uploader('Upload your CT scan', accept_multiple_files=False)
    if img is not None:
        with open(os.path.join(r"C:\Users\hp\Desktop\Review-3\lung_cancer_disease\user_uploaded_images","userScan.jpg"),"wb") as f: 
            f.write(img.getbuffer())
        st.image(img,caption='your CT Scan',width=200)
        loaded_model = keras.models.load_model('C:/Users/hp/Desktop/Review-3/lung_cancer_disease/model/lung_cancer_detection.h5',compile=False)
        print("model loaded")
        path = "C:/Users/hp/Desktop/Review-3/lung_cancer_disease/user_uploaded_images/"
        testing = []
        IMG_SIZE = 180
        for images in os.listdir(path):
            img = cv2.imread(path+images)
            testing.append(cv2.resize(img,(IMG_SIZE,IMG_SIZE)))
        testing = np.asarray(testing)
        prediction = loaded_model.predict(testing)
        prediction = np.argmax(prediction, axis= 1)
        print("res = ",prediction)
        if prediction == 1 or prediction == 0:
                st.error("You either have lung cancer or likely to have it. Please visit the doctor as soon as possible.")
        else:
                st.success("Hurray! You don't have lung cancer.")


def brain_tumor_prediction():
    brain_tumor_bg_img = """ 
    <style>
        [data-testid="stAppViewContainer"]{
            background-image: url("https://t3.ftcdn.net/jpg/01/41/37/40/360_F_141374009_t96qWESc4wEgSiCmUBIXMMAuio3HaxqK.jpg");
            background-size: cover;
        }
        footer {
	    visibility: hidden;
	    }
        #MainMenu{
            visibility: hidden;
        }
        [data-testid="stHeader"] {
            display: none;
        } 
    </style>
    """
    st.markdown(brain_tumor_bg_img,unsafe_allow_html=True)
    img = st.file_uploader('Upload your CT scan', accept_multiple_files=False)
    if img is not None:
        with open(os.path.join(r"C:\Users\hp\Desktop\Review-3\brain_tumor_disease\user_uploaded_images","userScan.jpg"),"wb") as f: 
            f.write(img.getbuffer())
        st.image(img,caption='your CT Scan',width=200)
        path = "C:/Users/hp/Desktop/Review-3/brain_tumor_disease/user_uploaded_images/"
        testing = []
        for images in os.listdir(path):
            image = cv2.imread(path+images)
            image = cv2.resize(image, (224, 224))
            testing.append(image)
        loaded_model = keras.models.load_model('C:/Users/hp/Desktop/Review-3/brain_tumor_disease/brain_tumor_detection.h5',compile=False)
        batch_size = 64
        testing = np.array(testing) / 255.0
        prediction = loaded_model.predict(testing, batch_size= batch_size)
        prediction = np.argmax(prediction, axis= 1)
        print("res = ",prediction)
        if prediction == 1:
                st.error("You either have brain tumor problem or are likely to have it. Please visit the doctor as soon as possible.")
        else:
                st.success("Hurray! You don't have brain tumor.")

def main():
    st.header("Chronic Disease Prediction")
    options = ["Diabetes Prediction", "Heart Disease Prediction", "Lung Cancer Prediction", "Brain Tumor Prediction"]
    choice = st.selectbox("Select a disease to predict:", options)

    if choice == "Diabetes Prediction":
        diabetes_prediction()
    elif choice == "Heart Disease Prediction":
        heart_disease_prediction()
    elif choice == "Lung Cancer Prediction":
        lung_cancer_prediction()
    else:
        brain_tumor_prediction()

if __name__=='__main__':
    main()

