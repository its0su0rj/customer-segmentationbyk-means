# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 18:22:11 2024

@author: sujee
"""

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the fitted scaler
model = joblib.load('customer_segmentation_model.joblib')
scaler = joblib.load('scaler.joblib')

# Create a sidebar for user input
st.sidebar.header('Customer Segmentation')
quantity = st.sidebar.number_input('Quantity')
unit_price = st.sidebar.number_input('Unit Price')

# Create a DataFrame from the input data
data = pd.DataFrame([[quantity, unit_price]], columns=['Quantity', 'UnitPrice'])

# Standardize the input data
data_scaled = scaler.transform(data)

# Make predictions on the input data
segment = model.predict(data_scaled)

# Display the result
st.write(f'The customer belongs to segment {segment[0]}')
