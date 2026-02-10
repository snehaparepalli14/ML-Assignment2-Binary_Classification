import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from models.utils import evaluate_model


# Ensure model save directory exists

MODEL_DIR = "models/saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# 1. Load Dataset

DATA_PATH = "data/data.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# 2. Drop unnecessary columns

df.drop(columns=["id", "Unnamed: 32"], inplace=True)


# 3. Encode target variable

label_encoder = LabelEncoder()
df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])
# M -> 1, B -> 0


# 4. Split features and target

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# 5. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


# 6. Feature Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing completed successfully!")

# 7. Initialize Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

results = {}


# 8. Train, Predict, Evaluate & Save

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    results[model_name] = metrics

    # Save model
    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name.replace(' ', '_')}.pkl"
    )
    joblib.dump(model, model_path)

    print("Metrics:", metrics)

# 9. Display Summary Table

results_df = pd.DataFrame(results).T
print("\nFinal Evaluation Summary:")
print(results_df)
