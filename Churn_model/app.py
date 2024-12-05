import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# load all the models
model = tf.keras.models.load_model('churn_model.h5')

# load the label encoder
with open('label_encoder_gender.pkl', 'rb') as f:
    le = pickle.load(f)

# load the standard scaler
with open('scaler.pkl', 'rb') as f:
    sc = pickle.load(f)

# load the one hot encoder
with open('onehot_encoder_geography.pkl', 'rb') as f:
    ohe = pickle.load(f)

# streamlit app
st.title('Customer Churn Prediction')
st.write('This is a simple app to predict customer churn using a neural network model')

# input form
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox("Gender", le.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', 0, 100)
num_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [le.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

# one hot encoding 'Geography'
encoded_geography = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    encoded_geography, columns=ohe.get_feature_names_out(['Geography']))

# combine the dataframes
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# standardize the data
input_data_scaled = sc.transform(input_data)

# make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write('Customer will churn')
else:
    st.write('Customer will not churn')
