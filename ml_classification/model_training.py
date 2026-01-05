# Trains losgistic regression based on data from data_acquisition.py

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Load my data from the feature extraction script
df = pd.read_csv('features.csv', names=['label', 'mean', 'std', 'rms', 'p2p', 'dom_freq', 'band_ratio'], header = 0)

# Separate the labels (what we want to predict) from the data (features)
X = df[['mean', 'std', 'rms', 'p2p', 'dom_freq', 'band_ratio']].values
y = df['label'].values

# I am dividing by the maximum to get everything between 0 and 1 
# so the big numbers don't overwhelm the small ones.
X_scaled = X / X.max(axis=0)

# Split into Training (80%) and Testing (20%)
# I used 'stratify' to make sure each LED frequency is in both groups
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

# Create and train the model
# I used Logistic Regression because it's a clear way to categorize data
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

print(f"My model's accuracy on new data: {score:.2%}")

# Look at the Confusion Matrix to see where the model got confused
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
