import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --- Page Configuration ---
st.set_page_config(
    page_title="HeartHealth AI",
    page_icon="💓",
    layout="centered"
)

# --- Custom Header ---
st.title("💓 HeartHealth AI: Predictive Analytics")
st.markdown("""
    This application uses a **K-Nearest Neighbors (KNN)** algorithm to assess the likelihood of heart disease 
    based on clinical parameters. 
    *Developed by [Your Name/TheRashi]*
""")

# --- Data Engine (Cached for Performance) ---
@st.cache_resource
def train_heart_model(csv_path):
    """Loads data and trains the KNN model."""
    df = pd.read_csv(csv_path)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Splitting
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Model Initialization
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    
    return knn_model, scaler

# Load model and scaler
try:
    model, scaler = train_heart_model("heart.csv")
except FileNotFoundError:
    st.error("Dataset 'heart.csv' not found. Please ensure it is in the same directory.")
    st.stop()

# --- User Interface: Inputs ---
st.sidebar.header("📋 Patient Clinical Data")

def get_input_data():
    with st.sidebar:
        # Categorical inputs with more descriptive labels
        age = st.slider("Age", 20, 90, 45)
        sex = st.radio("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                          help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False")
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        thalach = st.slider("Max Heart Rate (thalach)", 70, 210, 150)
        exang = st.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
        slope = st.selectbox("ST Segment Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    feature_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame([feature_dict])

user_df = get_input_data()

# --- Prediction Logic ---
st.divider()
if st.button("Analyze Results"):
    # Preprocessing
    scaled_data = scaler.transform(user_df)
    result = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)

    # Output display
    st.subheader("🔍 Clinical Analysis Result")
    
    if result[0] == 1:
        st.error(f"### High Risk Detected")
        st.write(f"Confidence Level: **{probability[0][1]*100:.1f}%**")
        st.warning("The model indicates indicators consistent with heart disease. Consult a medical professional.")
    else:
        st.success(f"### Low Risk Detected")
        st.write(f"Confidence Level: **{probability[0][0]*100:.1f}%**")
        st.info("The model suggests clinical parameters are within normal ranges for heart disease.")