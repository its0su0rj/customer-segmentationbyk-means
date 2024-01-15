# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 18:21:14 2024

@author: sujee
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load the customer data
df = pd.read_csv('online.csv')

# Preprocess the data
df = df.dropna()
df = df[['Quantity', 'UnitPrice']]

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Train the K-means model
model = KMeans(n_clusters=3)
model.fit(df_scaled)

# Save the trained model and the fitted scaler
joblib.dump(model, 'customer_segmentation_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
