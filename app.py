import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    auc
)

# -----------------------------
# App Configuration
# -----------------------------

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

st.title("Breast Cancer Classification – ML Models")
st.write("Upload test data, select a trained model, and evaluate performance.")

# -----------------------------
# Sample Test Dataset Download
# -----------------------------

st.subheader("Download Sample Test Dataset")

def create_sample_test_data():
    sample_data = {
        "radius_mean": [17.99, 13.71],
        "texture_mean": [10.38, 20.83],
        "perimeter_mean": [122.8, 90.2],
        "area_mean": [1001, 577.9],
        "smoothness_mean": [0.1184, 0.1189],
        "compactness_mean": [0.2776, 0.1645],
        "concavity_mean": [0.3001, 0.09366],
        "concave_points_mean": [0.1471, 0.05985],
        "symmetry_mean": [0.2419, 0.2196],
        "fractal_dimension_mean": [0.07871, 0.07451],
        "radius_se": [1.095, 0.5835],
        "texture_se": [0.9053, 1.377],
        "perimeter_se": [8.589, 3.856],
        "area_se": [153.4, 50.96],
        "smoothness_se": [0.006399, 0.008805],
        "compactness_se": [0.04904, 0.03029],
        "concavity_se": [0.05373, 0.02488],
        "concave_points_se": [0.01587, 0.01448],
        "symmetry_se": [0.03003, 0.01486],
        "fractal_dimension_se": [0.006193, 0.005412],
        "radius_worst": [25.38, 17.06],
        "texture_worst": [17.33, 28.14],
        "perimeter_worst": [184.6, 110.6],
        "area_worst": [2019, 897],
        "smoothness_worst": [0.1622, 0.1654],
        "compactness_worst": [0.6656, 0.3682],
        "concavity_worst": [0.7119, 0.2678],
        "concave_points_worst": [0.2654, 0.1556],
        "symmetry_worst": [0.4601, 0.3196],
        "fractal_dimension_worst": [0.1189, 0.1151],
        "diagnosis": ["M", "B"]
    }
    return pd.DataFrame(sample_data)

sample_df = create_sample_test_data()
sample_csv = sample_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Sample Test CSV",
    data=sample_csv,
    file_name="sample_test_dataset.csv",
    mime="text/csv"
)

st.divider()

# -----------------------------
# Model Configuration
# -----------------------------

MODEL_DIR = "models/saved_models"

MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a Model",
    list(MODEL_FILES.keys())
)

# -----------------------------
# Upload Dataset
# -----------------------------

uploaded_file = st.file_uploader("Upload CSV Test Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Preprocessing
    # -----------------------------

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    if "Unnamed: 32" in df.columns:
        df.drop(columns=["Unnamed: 32"], inplace=True)

    if "diagnosis" not in df.columns:
        st.error("Dataset must contain 'diagnosis' column.")
        st.stop()

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Load Selected Model
    # -----------------------------

    model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model_name])
    model = joblib.load(model_path)

    # -----------------------------
    # Prediction
    # -----------------------------

    if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

    # -----------------------------
    # Download Predictions
    # -----------------------------

    results_df = X.copy()
    results_df["Actual"] = y
    results_df["Predicted"] = y_pred
    results_df["Prediction_Probability"] = y_prob

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Predictions CSV",
        data=csv,
        file_name="model_predictions.csv",
        mime="text/csv"
    )

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------

    st.subheader(f"Evaluation Metrics – {selected_model_name}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col1.metric("Precision", round(precision_score(y, y_pred), 4))

    col2.metric("Recall", round(recall_score(y, y_pred), 4))
    col2.metric("F1 Score", round(f1_score(y, y_pred), 4))

    col3.metric("AUC", round(roc_auc_score(y, y_prob), 4))
    col3.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

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

    # -----------------------------
    # ROC Curve
    # -----------------------------

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()

    st.pyplot(fig2)

    # -----------------------------
    # Compare All Models
    # -----------------------------

    st.subheader("Performance Comparison Across All Models")

    if st.button("Compare All Models"):

        comparison_results = {}

        for name, file in MODEL_FILES.items():

            model_path = os.path.join(MODEL_DIR, file)
            model = joblib.load(model_path)

            if name in ["Logistic Regression", "KNN", "Naive Bayes"]:
                preds = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)[:, 1]
            else:
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]

            comparison_results[name] = {
                "Accuracy": accuracy_score(y, preds),
                "F1 Score": f1_score(y, preds),
                "AUC": roc_auc_score(y, probs)
            }

        comparison_df = pd.DataFrame(comparison_results).T

        st.dataframe(comparison_df)

        fig3, ax3 = plt.subplots()
        comparison_df["AUC"].plot(kind="bar", ax=ax3)
        ax3.set_ylabel("AUC Score")
        ax3.set_title("AUC Comparison Across Models")
        plt.xticks(rotation=45)

        st.pyplot(fig3)

        st.subheader("Observations")
        st.markdown("""
        - Ensemble models like Random Forest and XGBoost often show strong AUC performance.
        - Logistic Regression provides stable and interpretable results.
        - Models with higher AUC have better class separation ability.
        - In medical diagnosis, minimizing false negatives is critical.
        """)

else:
    st.info("Please upload a CSV test dataset to begin.")