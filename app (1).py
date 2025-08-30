import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset (PIMA Diabetes Dataset)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.title("🏥 Diabetes Prediction App")
st.write("This app predicts whether a person has diabetes based on medical parameters.")

# Sidebar user input
st.sidebar.header("Input Patient Data")

def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 20.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
    age = st.sidebar.slider("Age", 21, 100, 33)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Combine user input with dataset for consistency
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
input_scaled = scaler.transform(input_df)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display results
st.subheader("Patient Data")
st.write(input_df)

st.subheader("Prediction")
diabetes_result = np.array(["No Diabetes", "Diabetes"])
st.write(diabetes_result[prediction][0])

st.subheader("Prediction Probability")
st.write(prediction_proba)

# Model Accuracy
st.subheader("Model Accuracy on Test Set")
st.write(f"{accuracy_score(y_test, model.predict(X_test_scaled)) * 100:.2f}%")
