import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('water_potability.csv')

# Fill missing values with mean
data['ph'] = data['ph'].fillna(data['ph'].mean())
data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
data['Trihalomethanes'] = data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())

# Split the data
x = data.drop('Potability', axis=1)
y = data['Potability']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Train the model
reg = LogisticRegression()
reg.fit(x_train, y_train)

# Streamlit app
st.title("Water Potability Prediction App")

# Input features
ph = st.number_input("Enter pH:", min_value=0.0, max_value=14.0, value=7.0)
Hardness = st.number_input("Enter Hardness:", min_value=0.0, value=200.0)
Solids = st.number_input("Enter Solids:", min_value=0.0, value=20000.0)
Chloramines = st.number_input("Enter Chloramines:", min_value=0.0, value=5.0)
Sulfate = st.number_input("Enter Sulfate:", min_value=0.0, value=300.0)
Conductivity = st.number_input("Enter Conductivity:", min_value=0.0, value=400.0)
Organic_carbon = st.number_input("Enter Organic Carbon:", min_value=0.0, value=10.0)
Trihalomethanes = st.number_input("Enter Trihalomethanes:", min_value=0.0, value=30.0)
Turbidity = st.number_input("Enter Turbidity:", min_value=0.0, value=5.0)

# Predict button
if st.button("Predict Potability"):
    input_data = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
    prediction = reg.predict(input_data)

    # Display prediction
    st.success("Predicted Potability:")
    if prediction[0] == 1:
        st.write("Potable")
    else:
        st.write("Not Potable")
