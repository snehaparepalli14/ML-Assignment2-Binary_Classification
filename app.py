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

    st.write("Total Uploaded Rows:", df.shape[0])

    # -----------------------------
    # Download Uploaded Test Dataset
    # -----------------------------

    uploaded_csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Uploaded Test Dataset",
        data=uploaded_csv,
        file_name="uploaded_test_dataset.csv",
        mime="text/csv"
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------

    df_original = df.copy()

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

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # -----------------------------
    # Download Predictions
    # -----------------------------

    results_df = df_original.copy()
    results_df["Predicted"] = y_pred
    results_df["Prediction_Probability"] = y_prob

    st.write("Download Rows:", results_df.shape[0])

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

            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

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

        st.markdown("""
        **Observations:**
        - Ensemble models generally demonstrate stronger AUC performance.
        - Logistic Regression provides stable and interpretable baseline results.
        - Higher AUC indicates better class separation ability.
        - In medical diagnosis, minimizing false negatives (high recall) is critical.
        """)

else:
    st.info("Please upload a CSV test dataset to begin.")
