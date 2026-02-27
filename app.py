"""
Mental Stress Detection System - Web Application
==================================================
A professional Streamlit web application that predicts
mental stress levels using a trained Random Forest model.

Features:
- User-friendly input form for lifestyle data
- Real-time stress level prediction (Low / Medium / High)
- Visual stress meter using Plotly gauge chart
- Personalized recommendations based on stress level
- Model accuracy display

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Mental Stress Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS for Professional Styling
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.05rem;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #4a5568;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        color: #2d3748;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Result cards */
    .result-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 5px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Recommendation styling */
    .recommendation-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #718096;
        font-size: 0.85rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Model, Scaler, and Accuracy
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model and scaler from saved files."""
    model = joblib.load('stress_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_accuracy():
    """Load the model accuracy from saved file."""
    if os.path.exists('model_accuracy.txt'):
        with open('model_accuracy.txt', 'r') as f:
            return f.read().strip()
    return "N/A"

# Load resources
model, scaler = load_model()
accuracy = load_accuracy()

# ============================================
# Stress Level Labels and Colors
# ============================================
STRESS_LABELS = {0: "Low", 1: "Medium", 2: "High"}
STRESS_COLORS = {0: "#28a745", 1: "#ffc107", 2: "#dc3545"}
STRESS_EMOJIS = {0: "😊", 1: "😐", 2: "😰"}

# ============================================
# Helper Functions
# ============================================
def create_stress_gauge(stress_level, confidence):
    """
    Create a visual stress meter using Plotly gauge chart.

    Args:
        stress_level: 0 (Low), 1 (Medium), 2 (High)
        confidence: Model confidence percentage
    """
    # Map stress level to gauge value (0-100)
    gauge_value = stress_level * 50 + (confidence / 100) * 25

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_value,
        number={'suffix': '%', 'font': {'size': 40, 'color': '#2d3748'}},
        title={'text': "Stress Level Meter", 'font': {'size': 20, 'color': '#4a5568'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#4a5568"},
            'bar': {'color': STRESS_COLORS[stress_level], 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 33], 'color': '#d4edda'},    # Low - green
                {'range': [33, 66], 'color': '#fff3cd'},    # Medium - yellow
                {'range': [66, 100], 'color': '#f8d7da'}    # High - red
            ],
            'threshold': {
                'line': {'color': STRESS_COLORS[stress_level], 'width': 4},
                'thickness': 0.8,
                'value': gauge_value
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )

    return fig


def get_recommendations(stress_level):
    """
    Return personalized recommendations based on the predicted stress level.

    Args:
        stress_level: 0 (Low), 1 (Medium), 2 (High)

    Returns:
        Dictionary with category-wise recommendations
    """
    recommendations = {
        0: {  # Low Stress
            "title": "Your Stress Level is Low - Keep It Up!",
            "summary": "You're managing your stress well. Here are some tips to maintain your current state:",
            "tips": {
                "Lifestyle": [
                    "Continue your current sleep schedule - it's working well",
                    "Maintain your work-life balance",
                    "Keep up with your physical activities"
                ],
                "Mental Wellness": [
                    "Practice gratitude journaling to maintain positivity",
                    "Continue nurturing your social relationships",
                    "Consider mindfulness meditation to stay centered"
                ],
                "Prevention": [
                    "Set boundaries to avoid overcommitting at work",
                    "Schedule regular breaks during work hours",
                    "Plan leisure activities to look forward to"
                ]
            }
        },
        1: {  # Medium Stress
            "title": "Your Stress Level is Moderate - Take Action Now",
            "summary": "You're experiencing moderate stress. Here are actionable steps to reduce it:",
            "tips": {
                "Sleep & Rest": [
                    "Aim for 7-8 hours of quality sleep each night",
                    "Create a calming bedtime routine (no screens 1 hour before bed)",
                    "Try relaxation techniques like deep breathing before sleep"
                ],
                "Work Management": [
                    "Break large tasks into smaller, manageable chunks",
                    "Learn to say 'no' to additional commitments when overwhelmed",
                    "Take a 5-minute break every hour during work"
                ],
                "Physical Health": [
                    "Exercise for at least 30 minutes daily (walking, yoga, or gym)",
                    "Stay hydrated - drink at least 8 glasses of water daily",
                    "Reduce caffeine and sugar intake"
                ],
                "Social Support": [
                    "Talk to a friend or family member about how you feel",
                    "Join a hobby group or community activity",
                    "Consider professional counseling if stress persists"
                ]
            }
        },
        2: {  # High Stress
            "title": "Your Stress Level is High - Immediate Action Recommended",
            "summary": "You're experiencing high stress levels. Please prioritize your well-being with these steps:",
            "tips": {
                "Immediate Relief": [
                    "Practice the 4-7-8 breathing technique: Inhale 4s, Hold 7s, Exhale 8s",
                    "Take a 10-minute walk outside right now",
                    "Progressive muscle relaxation - tense and release each muscle group"
                ],
                "Sleep Priority": [
                    "Make sleep your top priority - aim for 8+ hours",
                    "Use guided sleep meditations (apps like Calm or Headspace)",
                    "Avoid work emails and screens at least 2 hours before bed"
                ],
                "Work Changes": [
                    "Speak with your manager about workload adjustments",
                    "Delegate tasks wherever possible",
                    "Set firm boundaries - no work beyond your scheduled hours"
                ],
                "Professional Help": [
                    "Consider speaking with a mental health professional",
                    "Explore Employee Assistance Programs (EAP) if available",
                    "Call a mental health helpline if you feel overwhelmed"
                ],
                "Daily Habits": [
                    "Start with just 10 minutes of meditation daily",
                    "Reduce or eliminate alcohol and caffeine",
                    "Connect with supportive friends or family every day"
                ]
            }
        }
    }

    return recommendations[stress_level]


# ============================================
# Sidebar - Information Panel
# ============================================
with st.sidebar:
    st.markdown("## About This System")
    st.markdown("""
    This AI-powered system uses a **Random Forest** machine learning model
    to predict your mental stress level based on lifestyle factors.
    """)

    st.markdown("---")
    st.markdown("### Model Information")
    st.markdown(f"**Algorithm:** Random Forest Classifier")
    st.markdown(f"**Accuracy:** {accuracy}%")
    st.markdown(f"**Training Data:** 1000 samples")
    st.markdown(f"**Features:** 6 lifestyle factors")

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. Enter your lifestyle data
    2. The AI model analyzes your inputs
    3. Predicts your stress level
    4. Provides personalized tips
    """)

    st.markdown("---")
    st.markdown("### Disclaimer")
    st.markdown("""
    *This tool is for educational purposes only.
    It is not a substitute for professional
    medical advice. If you're experiencing
    severe stress, please consult a
    healthcare professional.*
    """)

# ============================================
# Main Content - Header
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🧠 Mental Stress Detection System</h1>
    <p>AI-powered analysis of your lifestyle factors to assess stress levels</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# Quick Stats Row
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Model Type</h3>
        <div class="value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Model Accuracy</h3>
        <div class="value">{accuracy}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Training Samples</h3>
        <div class="value">1,000</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>Features Analyzed</h3>
        <div class="value">6</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Input Form Section
# ============================================
st.markdown("---")
st.markdown("### 📋 Enter Your Lifestyle Information")
st.markdown("*Fill in the details below to get your stress level prediction*")

# Create two columns for the input form
input_col1, input_col2 = st.columns(2)

with input_col1:
    # Age input
    age = st.slider(
        "🎂 Age",
        min_value=18,
        max_value=65,
        value=30,
        help="Select your current age (18-65 years)"
    )

    # Sleep hours input
    sleep_hours = st.slider(
        "😴 Sleep Hours (per day)",
        min_value=3.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        help="Average hours of sleep you get per day"
    )

    # Work hours input
    work_hours = st.slider(
        "💼 Work Hours (per day)",
        min_value=4.0,
        max_value=16.0,
        value=8.0,
        step=0.5,
        help="Average hours you spend working per day"
    )

with input_col2:
    # Physical activity level
    physical_activity = st.slider(
        "🏃 Physical Activity Level",
        min_value=1,
        max_value=10,
        value=5,
        help="1 = Very Low, 10 = Very Active"
    )

    # Social interaction level
    social_interaction = st.slider(
        "👥 Social Interaction Level",
        min_value=1,
        max_value=10,
        value=5,
        help="1 = Very Isolated, 10 = Very Social"
    )

    # Anxiety level
    anxiety_level = st.slider(
        "😟 Anxiety Level",
        min_value=1,
        max_value=10,
        value=5,
        help="1 = Very Calm, 10 = Very Anxious"
    )

# ============================================
# Prediction Section
# ============================================
st.markdown("---")

# Center the predict button
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    predict_button = st.button(
        "🔍 Analyze My Stress Level",
        use_container_width=True,
        type="primary"
    )

if predict_button:
    # Prepare input data as a numpy array
    # Must match the order of features used during training:
    # [age, sleep_hours, work_hours, physical_activity, social_interaction, anxiety_level]
    input_data = np.array([[age, sleep_hours, work_hours,
                            physical_activity, social_interaction, anxiety_level]])

    # Scale the input using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get prediction probabilities for confidence score
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = probabilities[prediction] * 100

    # Get stress label, color, and emoji
    stress_label = STRESS_LABELS[prediction]
    stress_color = STRESS_COLORS[prediction]
    stress_emoji = STRESS_EMOJIS[prediction]

    # ============================================
    # Display Results
    # ============================================
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # Result and Gauge in two columns
    result_col, gauge_col = st.columns([1, 1])

    with result_col:
        # Stress level result card
        css_class = f"result-{stress_label.lower()}"
        st.markdown(f"""
        <div class="{css_class}">
            <h2 style="margin: 0; color: {stress_color};">
                {stress_emoji} {stress_label} Stress
            </h2>
            <p style="margin: 0.5rem 0 0 0; color: #4a5568; font-size: 1.1rem;">
                Model Confidence: <strong>{confidence:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show probability breakdown
        st.markdown("#### Probability Breakdown")
        for i, (label, prob) in enumerate(zip(STRESS_LABELS.values(), probabilities)):
            emoji = STRESS_EMOJIS[i]
            st.progress(prob, text=f"{emoji} {label}: {prob*100:.1f}%")

    with gauge_col:
        # Display the stress gauge chart
        gauge_fig = create_stress_gauge(prediction, confidence)
        st.plotly_chart(gauge_fig, use_container_width=True)

    # ============================================
    # Input Summary
    # ============================================
    st.markdown("#### 📝 Your Input Summary")
    summary_cols = st.columns(6)
    inputs_display = [
        ("Age", f"{age} yrs"),
        ("Sleep", f"{sleep_hours} hrs"),
        ("Work", f"{work_hours} hrs"),
        ("Activity", f"{physical_activity}/10"),
        ("Social", f"{social_interaction}/10"),
        ("Anxiety", f"{anxiety_level}/10")
    ]
    for col, (label, value) in zip(summary_cols, inputs_display):
        with col:
            st.metric(label=label, value=value)

    # ============================================
    # Personalized Recommendations
    # ============================================
    st.markdown("---")
    recs = get_recommendations(prediction)

    st.markdown(f"### 💡 {recs['title']}")
    st.markdown(f"*{recs['summary']}*")

    # Display recommendations in organized columns
    rec_categories = list(recs['tips'].items())
    num_cols = min(len(rec_categories), 3)

    for row_start in range(0, len(rec_categories), num_cols):
        cols = st.columns(num_cols)
        for idx, col in enumerate(cols):
            cat_idx = row_start + idx
            if cat_idx < len(rec_categories):
                category, tips = rec_categories[cat_idx]
                with col:
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h4 style="color: #4a5568; margin-top: 0;">📌 {category}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for tip in tips:
                        st.markdown(f"- {tip}")

# ============================================
# Footer
# ============================================
st.markdown("""
<div class="footer">
    <p>Mental Stress Detection System | Built with Streamlit & Scikit-Learn | CSE Mini Project</p>
    <p>⚠️ This is an educational project. Not a medical diagnostic tool.</p>
</div>
""", unsafe_allow_html=True)
