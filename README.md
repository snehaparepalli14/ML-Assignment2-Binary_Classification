# Machine Learning Assignment 2 – Breast Cancer Classification

## a. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether a breast tumor is malignant or benign based on diagnostic features. The project also includes deploying the trained models through an interactive Streamlit web application to demonstrate an end-to-end machine learning workflow including modeling, evaluation, and deployment.

---

## b. Dataset Description

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, sourced from the UCI Machine Learning Repository via Kaggle.

- Number of instances: 569  
- Number of features: 30 numerical features  
- Target variable: `diagnosis`  
  - `M` → Malignant  
  - `B` → Benign  

The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. These features describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

Data preprocessing steps included removal of non-informative columns, label encoding of the target variable, stratified train-test split, and feature scaling where required.

---

## c. Models Used and Evaluation Metrics

The following six classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

The evaluation metrics used were:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbors | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost (Ensemble) | 0.9737 | 0.9940 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

---

## d. Observations on Model Performance

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Demonstrated strong baseline performance with high accuracy and excellent AUC, indicating good linear separability of the dataset. |
| Decision Tree | Provided interpretable results but showed lower performance due to overfitting compared to ensemble methods. |
| K-Nearest Neighbors | Achieved high accuracy after feature scaling, showing effective use of distance-based classification. |
| Naive Bayes | Performed reasonably well with high AUC, though its independence assumption slightly reduced accuracy and recall. |
| Random Forest (Ensemble) | Delivered one of the best performances with perfect precision and high MCC, benefiting from ensemble averaging. |
| XGBoost (Ensemble) | Achieved the strongest overall performance, combining high accuracy, AUC, and MCC, demonstrating robustness and generalization ability. |
