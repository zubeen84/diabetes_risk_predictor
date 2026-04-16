# Diabetes Risk Predictor
### End-to-end machine learning application with live deployment

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)
![Models](https://img.shields.io/badge/Models-6%20ML%20Algorithms-blue)
![Responsible AI](https://img.shields.io/badge/Design-ISO%2042001%20Aligned-742774?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Live App

[Open the live application](https://diabetesriskpredictor-b85jnngpyqkpvbkmeac2q6.streamlit.app/)

---

## At a Glance

| | |
|---|---|
| **Problem** | Early diabetes detection using clinical measurements to support medical screening |
| **Approach** | Six ML models trained, evaluated, compared and deployed as an interactive app |
| **Best model** | Voting Ensemble - 88.9% recall, 75% accuracy |
| **Key challenge** | Class imbalance in medical data, addressed using IQR filtering, log transformation and SMOTE |
| **Deployment** | Live Streamlit app - accessible on desktop and mobile |
| **Governance** | ISO 42001 aligned responsible AI disclaimer built into the interface |

---

## Overview

A machine learning web application that predicts the probability of a patient 
developing diabetes within five years, using the Pima Indians Diabetes Database 
(UCI ML Repository).

This project demonstrates a complete data science workflow: data preprocessing, 
handling class imbalance, training and evaluating six ML models, selecting the 
best performer, and deploying a live interactive application.

**Why recall was prioritised over accuracy:**
In a medical screening context, a false negative, predicting no diabetes when 
the condition is present, is more harmful than a false positive. Model selection 
therefore prioritised recall (sensitivity) to minimise missed cases, not raw accuracy. 
The Voting Ensemble achieved 88.9% recall, the strongest result across all models tested.

---

## Model Comparison

| Model | Recall | Accuracy | Notes |
|-------|--------|----------|-------|
| Voting Ensemble (recommended) | 88.9% | 75% | Best recall across all models |
| Logistic Regression | 70% | 74% | Most interpretable |
| Balanced Random Forest | 64% | 74% | Handles class imbalance well |
| Random Forest | 57% | 76% | Best specificity |
| XGBoost | 58% | 73% | Handles complex non-linear patterns |
| SVC | 52% | 74% | Strong decision boundary detection |

All six models are selectable in the live app, allowing direct comparison of 
predictions and performance characteristics.

---

## Application Features

- Enter 8 clinical measurements via interactive sliders
- Select from 6 trained ML models including Voting Ensemble
- Risk score displayed as a percentage with colour-coded risk levels
- Model performance comparison table in sidebar for transparency
- ISO 42001 aligned responsible AI disclaimer built into the interface
- Mobile-friendly responsive layout

---

## Dataset

| | |
|---|---|
| **Source** | Pima Indians Diabetes Database - UCI ML Repository |
| **Size** | 768 patients, 8 clinical features |
| **Target** | Binary classification - Diabetic (1) / Non-Diabetic (0) |
| **Class imbalance** | Addressed using IQR filtering, log transformation, and SMOTE oversampling |

---

## Input Features

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| Diabetes Pedigree Function | Genetic risk score |
| Age | Age in years |

---

## Run Locally

```bash
git clone https://github.com/zubeen84/diabetes-risk-predictor.git
cd diabetes-risk-predictor
pip install -r requirements.requirements
streamlit run diabetes_app.py
```

---

## Related Project

[Full data science notebook](https://github.com/zubeen84/Machine-Learning-Model-for-Diabetes-Prediction) 
- covers full EDA, preprocessing pipeline, model training, evaluation and selection.

---

## Skills Demonstrated

`Python` `Scikit-learn` `XGBoost` `Streamlit` `Machine Learning` `Classification`  
`Ensemble Methods` `Logistic Regression` `Random Forest` `SVC` `SMOTE`  
`Class Imbalance Handling` `Model Evaluation` `Recall Optimisation` `EDA`  
`Feature Engineering` `Data Preprocessing` `Model Deployment` `Responsible AI`  
`ISO 42001` `Healthcare Analytics` `UCI ML Repository`

---

## Responsible AI

This application is for informational and educational purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Built in alignment with ISO 42001 AI Management System principles including 
transparency of model limitations, human oversight, and appropriate use disclaimers.

---

## Author

**Zubeen Khalid**
MSc Applied Data Science - Anglia Ruskin University
ISO 42001 Certified | AI+ Foundation | Prompt Engineering Level 1

[LinkedIn](https://www.linkedin.com/in/zubeenkhalid) · [GitHub](https://github.com/zubeen84)
