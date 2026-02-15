# Comparative Analysis of Machine Learning Algorithms for Breast Cancer Classification

## 1. Executive Summary

The objective of this project is to design, implement, and evaluate a
machine learning framework for the classification of breast tumors as
Malignant or Benign using the Breast Cancer Wisconsin Diagnostic
dataset.

Six supervised machine learning models were trained and evaluated using
standardized preprocessing techniques. The final deliverable includes a
deployed Streamlit web application that enables dataset upload, model
selection, performance evaluation, and downloadable predictions.

------------------------------------------------------------------------

## 2. Dataset Methodology

Source: Breast Cancer Wisconsin Diagnostic Dataset\
Problem Type: Binary Classification\
Target Variable: diagnosis (0 = Benign, 1 = Malignant)

### Dataset Characteristics

-   Total Samples: 569\
-   Total Features: 30 numerical features\
-   Target Classes: Malignant and Benign

### Preprocessing Steps

-   Removed unnecessary identifier columns\
-   Encoded target labels (M=1, B=0)\
-   Applied StandardScaler for feature scaling\
-   Used Stratified 80/20 Train-Test Split

------------------------------------------------------------------------

## 3. Experimental Results

  Model                 Accuracy   Precision   ROC-AUC
  --------------------- ---------- ----------- ---------
  Logistic Regression   0.9825     0.9762      0.9970
  Decision Tree         0.9298     0.9048      0.9246
  K-Nearest Neighbors   0.9561     0.9383      0.9830
  Naive Bayes           0.9211     0.8966      0.9878
  Random Forest         0.9737     0.9630      0.9929
  XGBoost               0.9737     0.9630      0.9940

------------------------------------------------------------------------

## 4. Analysis and Observations

-   Logistic Regression achieved the highest overall performance with
    98.25% accuracy and ROC-AUC of 0.9970.
-   Decision Tree showed comparatively lower performance and potential
    overfitting.
-   KNN performed strongly after feature scaling, confirming meaningful
    feature distance relationships.
-   Naive Bayes achieved high ROC-AUC despite slightly lower precision.
-   Random Forest and XGBoost demonstrated strong ensemble performance
    with ROC-AUC above 0.99.
-   Logistic Regression emerged as the best-performing model overall.

------------------------------------------------------------------------

## 5. Streamlit Application Features

-   CSV test dataset upload\
-   Model selection dropdown\
-   ROC curve visualization\
-   Confusion matrix visualization\
-   Performance comparison across models\
-   Download predictions as CSV

------------------------------------------------------------------------

## 6. Project Architecture

├── app.py\
├── models/\
│ ├── train_models.py\
│ ├── utils.py\
│ └── saved_models/\
├── requirements.txt\
├── README.md\
└── .gitignore

------------------------------------------------------------------------

## 7. Installation & Execution

1.  Clone Repository: git clone `https://ml-assignment2-binary-classification.streamlit.app`{=html}

2.  Install Dependencies: pip install -r requirements.txt

3.  Run Application: streamlit run app.py

Application runs at: http://localhost:8501

------------------------------------------------------------------------

## 8. Deployment

Platform: Streamlit Community Cloud  
Entry File: app.py  

Live Application Link:  
https://ml-assignment2-binary-classification.streamlit.app
------------------------------------------------------------------------

## 9. Conclusion

This project demonstrates a complete machine learning workflow including
preprocessing, training, evaluation, comparison, and deployment.
Logistic Regression provided the best overall classification performance
for this dataset.
