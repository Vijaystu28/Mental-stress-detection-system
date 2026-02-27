"""
Mental Stress Detection - Model Training Script
=================================================
This script trains a Random Forest Classifier to predict
stress levels (Low, Medium, High) based on user lifestyle data.

Steps:
1. Load the stress dataset
2. Preprocess the data
3. Split into training and testing sets
4. Apply feature scaling
5. Train the Random Forest model
6. Evaluate model accuracy
7. Save the trained model and scaler
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ============================================
# Step 1: Load the Dataset
# ============================================
print("=" * 50)
print("Mental Stress Detection - Model Training")
print("=" * 50)

# Get the path to the dataset
dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'stress_data.csv')

# Load CSV file into a pandas DataFrame
data = pd.read_csv(dataset_path)
print(f"\n[Step 1] Dataset loaded successfully!")
print(f"  Shape: {data.shape[0]} rows x {data.shape[1]} columns")
print(f"  Columns: {list(data.columns)}")

# ============================================
# Step 2: Data Preprocessing
# ============================================
print(f"\n[Step 2] Preprocessing data...")

# Check for missing values
missing = data.isnull().sum().sum()
print(f"  Missing values: {missing}")

# If there are missing values, fill them with column means
if missing > 0:
    data = data.fillna(data.mean())
    print("  Missing values filled with column means.")

# Separate features (X) and target variable (y)
# Features: all columns except 'stress_level'
X = data.drop('stress_level', axis=1)

# Target: the 'stress_level' column (0=Low, 1=Medium, 2=High)
y = data['stress_level']

print(f"  Features shape: {X.shape}")
print(f"  Target distribution:")
for level, count in y.value_counts().sort_index().items():
    label = ['Low', 'Medium', 'High'][level]
    print(f"    {label} ({level}): {count} samples ({count/len(y)*100:.1f}%)")

# ============================================
# Step 3: Train-Test Split
# ============================================
print(f"\n[Step 3] Splitting data into training and testing sets...")

# Split: 80% training, 20% testing
# random_state=42 ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training samples: {X_train.shape[0]}")
print(f"  Testing samples:  {X_test.shape[0]}")

# ============================================
# Step 4: Feature Scaling
# ============================================
print(f"\n[Step 4] Applying feature scaling (StandardScaler)...")

# StandardScaler normalizes features to have mean=0 and std=1
# This improves model performance by putting all features on the same scale
scaler = StandardScaler()

# Fit the scaler on training data and transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Scaling completed. Features normalized to zero mean and unit variance.")

# ============================================
# Step 5: Train the Random Forest Model
# ============================================
print(f"\n[Step 5] Training Random Forest Classifier...")

# Random Forest: an ensemble of 100 decision trees
# Each tree votes on the prediction, majority wins
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    max_depth=10,          # Maximum depth of each tree
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores for faster training
)

# Train the model on scaled training data
model.fit(X_train_scaled, y_train)
print(f"  Model trained with {model.n_estimators} trees.")

# ============================================
# Step 6: Evaluate Model Accuracy
# ============================================
print(f"\n[Step 6] Evaluating model performance...")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n  Model Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print(f"\n  Classification Report:")
target_names = ['Low Stress', 'Medium Stress', 'High Stress']
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

# Feature importance - shows which features matter most
print(f"  Feature Importance:")
feature_names = X.columns.tolist()
importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"    {name}: {imp:.4f}")

# ============================================
# Step 7: Save the Trained Model and Scaler
# ============================================
print(f"\n[Step 7] Saving model and scaler...")

# Save to the project root directory
root_dir = os.path.dirname(os.path.dirname(__file__))

# Save the trained model using joblib
model_path = os.path.join(root_dir, 'stress_model.pkl')
joblib.dump(model, model_path)
print(f"  Model saved to: {model_path}")

# Save the scaler (needed to scale new inputs during prediction)
scaler_path = os.path.join(root_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"  Scaler saved to: {scaler_path}")

# Save accuracy for display in the web app
accuracy_path = os.path.join(root_dir, 'model_accuracy.txt')
with open(accuracy_path, 'w') as f:
    f.write(f"{accuracy * 100:.2f}")
print(f"  Accuracy saved to: {accuracy_path}")

print(f"\n{'=' * 50}")
print(f"Training Complete! Model is ready for predictions.")
print(f"{'=' * 50}")
