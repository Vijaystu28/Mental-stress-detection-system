# Mental Stress Detection System

## Overview
An AI-powered Mental Stress Detection System built with Python, Streamlit, and Scikit-Learn. The system uses a Random Forest Classifier trained on synthetic lifestyle data to predict stress levels (Low, Medium, High) based on user inputs.

## Project Structure
```
/dataset/
  stress_data.csv          - Synthetic stress dataset (1000 samples)
  generate_dataset.py      - Script to generate the dataset
/model/
  train_model.py           - Model training script
app.py                     - Streamlit web application (main entry point)
stress_model.pkl           - Trained Random Forest model
scaler.pkl                 - Fitted StandardScaler for feature scaling
model_accuracy.txt         - Saved model accuracy score
```

## Tech Stack
- **Frontend:** Streamlit with custom CSS styling
- **ML:** Scikit-Learn (Random Forest Classifier)
- **Visualization:** Plotly (gauge chart for stress meter)
- **Data:** Pandas, NumPy

## How to Run
- The app runs via: `streamlit run app.py --server.port 5000`
- To retrain the model: `python model/train_model.py`
- To regenerate dataset: `python dataset/generate_dataset.py`

## Features
- 6 lifestyle input features (age, sleep, work hours, physical activity, social interaction, anxiety)
- Stress level prediction with confidence scores
- Visual stress meter gauge
- Personalized recommendations based on stress level
- Professional UI with gradient styling

## Model Details
- Algorithm: Random Forest Classifier (100 trees, max_depth=10)
- Accuracy: ~79%
- Features are scaled using StandardScaler before prediction
