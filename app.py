import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the fitted scaler
model = joblib.load('customer_segmentation_model.joblib')
scaler = joblib.load('scaler.joblib')

# Title
st.title('Customer Segmentation')

# User input for Quantity
quantity = st.number_input('Enter Quantity:', value=0)

# User input for Unit Price
unit_price = st.number_input('Enter Unit Price:', value=0)

# Create a DataFrame from the input data
data = pd.DataFrame([[quantity, unit_price]], columns=['Quantity', 'UnitPrice'])

# Standardize the input data
data_scaled = scaler.transform(data)

# Create a button to trigger predictions
if st.button('Predict Segment'):
    # Make predictions on the input data
    segment = model.predict(data_scaled)
    
    # Display the result
    st.write(f'The customer belongs to segment {segment[0]}')
