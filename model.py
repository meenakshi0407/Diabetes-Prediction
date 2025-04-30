import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load data
data = pd.read_csv('diabetes.csv')

st.title("Diabetes Prediction App")

target_column = 'Outcome'
X = data.drop(target_column, axis=1)
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_scaled=ss.fit_transform(X_train)
x_test_scaled=ss.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_scaled, y_train)
    
#Input values from user
selected_features = st.sidebar.title("Enter the input Values")
st.sidebar.write("Please enter the following details:")
Pregnancies = st.sidebar.text_input("Pregnancies", "0")
Glucose = st.sidebar.text_input("Glucose", "0")
BloodPressure = st.sidebar.text_input("BloodPressure", "0")
SkinThickness = st.sidebar.text_input("SkinThickness", "0")
Insulin = st.sidebar.text_input("Insulin", "0")
BMI = st.sidebar.text_input("BMI", "0")
DiabetesPedigreeFunction = st.sidebar.text_input("DiabetesPedigreeFunction", "0")
Age = st.sidebar.text_input("Age", "0")

input_data = np.array([[float(Pregnancies), float(Glucose), float(BloodPressure),float(SkinThickness), float(Insulin), float(BMI),float(DiabetesPedigreeFunction), float(Age)]])

input_data_scaled = ss.transform(input_data)
prediction = model.predict(input_data_scaled)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, model.predict(x_test_scaled))
#st.write("Model Accuracy:", accuracy * 100, "%")


class_mapping = {
    0: "Non-diabetic",
    1: "Diabetic"
}
Predicted_class = class_mapping[prediction[0]]

st.write("Predicted class:", Predicted_class)
st.write("The person have diabetes :/ " if prediction[0] == 1 else "The person does not have diabetes :)")
