"""
Generate Synthetic Stress Dataset
==================================
This script creates a realistic synthetic dataset for training
the Mental Stress Detection model. It generates 1000 samples
with features that correlate logically with stress levels.
"""

import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
n_samples = 1000

# Generate features with realistic distributions
# Age: between 18 and 65
age = np.random.randint(18, 66, n_samples)

# Sleep hours: between 3 and 10 (less sleep = more stress)
sleep_hours = np.round(np.random.uniform(3, 10, n_samples), 1)

# Work hours: between 4 and 16 (more work = more stress)
work_hours = np.round(np.random.uniform(4, 16, n_samples), 1)

# Physical activity level: 1-10 (less activity = more stress)
physical_activity = np.random.randint(1, 11, n_samples)

# Social interaction level: 1-10 (less interaction = more stress)
social_interaction = np.random.randint(1, 11, n_samples)

# Anxiety level: 1-10 (more anxiety = more stress)
anxiety_level = np.random.randint(1, 11, n_samples)

# Calculate stress score based on weighted combination of features
# Higher score = higher stress
stress_score = (
    (10 - sleep_hours) * 2.0       # Less sleep -> more stress
    + work_hours * 1.5              # More work -> more stress
    + (10 - physical_activity) * 1.0  # Less activity -> more stress
    + (10 - social_interaction) * 0.8  # Less social -> more stress
    + anxiety_level * 2.5            # More anxiety -> more stress
    + np.random.normal(0, 3, n_samples)  # Random noise for realism
)

# Normalize stress score to determine stress level categories
# Using percentile-based thresholds for balanced classes
p33 = np.percentile(stress_score, 33)
p66 = np.percentile(stress_score, 66)

stress_level = np.where(
    stress_score < p33, 0,           # 0 = Low Stress
    np.where(stress_score < p66, 1,  # 1 = Medium Stress
             2)                       # 2 = High Stress
)

# Create DataFrame with all features and target variable
data = pd.DataFrame({
    'age': age,
    'sleep_hours': sleep_hours,
    'work_hours': work_hours,
    'physical_activity': physical_activity,
    'social_interaction': social_interaction,
    'anxiety_level': anxiety_level,
    'stress_level': stress_level
})

# Save dataset to CSV file
output_path = os.path.join(os.path.dirname(__file__), 'stress_data.csv')
data.to_csv(output_path, index=False)

print(f"Dataset generated successfully!")
print(f"Total samples: {n_samples}")
print(f"Saved to: {output_path}")
print(f"\nStress Level Distribution:")
print(f"  Low (0):    {(stress_level == 0).sum()} samples")
print(f"  Medium (1): {(stress_level == 1).sum()} samples")
print(f"  High (2):   {(stress_level == 2).sum()} samples")
print(f"\nDataset Preview:")
print(data.head(10).to_string(index=False))
