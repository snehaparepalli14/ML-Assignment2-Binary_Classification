import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# App Config

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

st.title("Breast Cancer Classification – ML Models")
st.write("Upload test data, select a trained model, and view predictions & metrics.")


# Load Models

MODEL_DIR = "models/saved_models"

MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

# Sidebar

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a model",
    list(MODEL_FILES.keys())
)


# Dataset Upload

uploaded_file = st.file_uploader("Upload CSV Test Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Preprocessing
 
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    if "Unnamed: 32" in df.columns:
        df.drop(columns=["Unnamed: 32"], inplace=True)

    if "diagnosis" not in df.columns:
        st.error("Uploaded dataset must contain 'diagnosis' column.")
        st.stop()

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # Load Model
  
    model_path = os.path.join(
        MODEL_DIR, MODEL_FILES[selected_model_name]
    )
    model = joblib.load(model_path)


    # Prediction

    if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]


    # Metrics

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col1.metric("Precision", round(precision_score(y, y_pred), 4))

    col2.metric("Recall", round(recall_score(y, y_pred), 4))
    col2.metric("F1 Score", round(f1_score(y, y_pred), 4))

    col3.metric("AUC", round(roc_auc_score(y, y_prob), 4))
    col3.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))


    # Confusion Matrix

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to proceed.")
